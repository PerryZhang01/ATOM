# SPDX-License-Identifier: MIT
"""What a KV transfer costs, and when it is therefore released.

One transfer is priced as `latency + bytes / bandwidth`, and the three terms
come from three different places on purpose.

**The bytes are exact.** A transfer moves a whole number of paged blocks, and
what one block costs is the geometry the engine was built with, not a
measurement -- so the byte count carries no error at all and nothing here
re-derives a KV layout.

**The bandwidth is a spec peak, and it is spent at the derate the spec
declares.** A datasheet number spent as an achievable one would make every
transfer faster than any wire has ever been. The schema obliges a derate
wherever a peak appears and refuses a document that omits one, so this module
reads both and multiplies; it re-checks nothing.

**The latency and the bandwidth are named per side of the node boundary**, and
the caller says which side it is on. There is no default and no fall back to
the other side: a transfer between two nodes priced on the intra-node link is
wrong by roughly an order of magnitude and would look entirely plausible.

Two things this model is not.

It is **unvalidated**. No real RDMA baseline has been fitted to it, so the
latency and bandwidth in a spec are declarations rather than measurements, and
a predicted transfer time inherits that. Fitting them is a calibration
exercise against a real deployment; until then the numbers are as good as the
document they came from and no better.

It charges the same duration to both ends of the same wire. A push and a pull
of the same blocks cost the same here, which is a claim about the fabric being
symmetric that this model cannot check.

Nothing in this module imports the engine or a tensor library. Reaching it
through the package around it is not the same import: `atom.compass.kv` loads
the connector beside it, and that brings the engine's connector interface, its
factory, and whatever those pull in.
None of it is a tensor library and none of it is a device runtime, so the
arithmetic still runs where no driver does -- which is the property worth
having, and it is not a bare import of this file.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Scope(enum.Enum):
    """Which side of the node boundary a transfer crosses.

    The spec carries a latency and a bandwidth for each, and they differ by
    enough that picking the wrong one is not a small error.
    """

    INTRA_NODE = "intra_node"
    INTER_NODE = "inter_node"


@dataclass(frozen=True, slots=True)
class TransferModel:
    """The price of one transfer: a fixed cost, a rate, and a block size.

    `bandwidth_bytes_per_s` is what a transfer actually reaches, not what the
    link is rated at -- `from_spec` applies the derate before it gets here, so
    a model built by hand in a test states an achievable number too.
    """

    latency_s: float
    bandwidth_bytes_per_s: float
    bytes_per_block: int

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError(f"latency_s must not be negative, got {self.latency_s}")
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError(
                "bandwidth_bytes_per_s must be positive, got "
                f"{self.bandwidth_bytes_per_s}"
            )
        if self.bytes_per_block < 1:
            raise ValueError(
                f"bytes_per_block must be at least 1, got {self.bytes_per_block}"
            )

    @classmethod
    def from_spec(cls, spec, geometry, scope: Scope) -> TransferModel:
        """Price a block of *geometry* on the *scope* link of *spec*.

        The spec refuses a link it does not carry, and refuses a peak
        bandwidth whose derate is missing; both refusals travel out of here
        unchanged, naming the field and the rule that declined it.
        """
        link = f"interconnect.{scope.value}"
        peak = spec.value(f"{link}.link_bandwidth_bytes_per_s")
        derate = spec.value(f"{link}.derate")
        return cls(
            latency_s=spec.value(f"{link}.link_latency_s"),
            bandwidth_bytes_per_s=peak * derate,
            bytes_per_block=geometry.bytes_per_block,
        )

    def duration_s(self, blocks: int) -> float:
        """How long *blocks* take on this link.

        Zero blocks is a real case -- a consumer whose prefix cache already
        holds the whole prompt still has a transfer to complete -- and it
        costs the latency, because the two ends still have to say so.
        """
        if blocks < 0:
            raise ValueError(
                f"a transfer of {blocks} blocks is not a transfer; a block "
                "count below zero means the computed-block count exceeded the "
                "allocation, which would price a transfer that finishes "
                "before it starts"
            )
        return (
            self.latency_s
            + (blocks * self.bytes_per_block) / self.bandwidth_bytes_per_s
        )

    def release_at(self, issue_time_s: float, blocks: int) -> float:
        """The time this transfer may be reported finished, and not before."""
        return issue_time_s + self.duration_s(blocks)

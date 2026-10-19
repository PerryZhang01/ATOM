# SPDX-License-Identifier: MIT
"""The store: a directory per entry, published once and never again.

**The store's physical form is a directory convention, with no index and no
manifest.** A key renders to a path by one total function, and the entry
restates its key, its topology, its provenance and a digest of every member
inside `entry.json`. Resolution is then arithmetic over the key rather than a
lookup, and there is no second record that can disagree with the first. The
eight incidents this package exists for are each a second statement of one fact
drifting away from it; an index is one more of those, and it is the one a store
would have to maintain on every write, so there is none. The refusal to
overwrite comes from the filesystem for free: an entry is built elsewhere and
moved into place, and a published entry is never empty -- it always carries
`entry.json` -- so `rename(2)` cannot replace one.

Stated that way on purpose, because the obvious stronger claim is false, as
measured: on ext4, `os.rename` **succeeds** onto an existing *empty* directory
and fails with ENOTEMPTY only onto a non-empty one.
The exclusivity is a consequence of what a published entry contains, not of the
rename primitive refusing an existing name -- so an entry form with no members
would inherit a silent overwrite, and an empty directory sitting in the way is
caught by the occupancy check before the rename, not by the rename. Staging is
per-publisher (`mkdtemp`) rather than a function of the key, so two publishers
at one key cannot write into each other's half-built entry.

Three refusals the convention buys, each one of those incidents:

**A handed-off entry cannot be quietly replaced.** Publishing to a key that
already has an entry is refused, and the refusal states the stored digest and
whether the incoming members are the same ones -- so a notes-only regeneration,
whose payload is bit-identical and whose digest is not, is visibly the thing it
is. The notes sit inside the digested document precisely so that changing them
changes the identity.

**Every rank of the topology is present, or nothing is published.** The
four-ranks-one-file incident wrote one name four times and left a survivor that
looked complete at 807 operators. Here a publish states its topology and its
members are checked against every rank of it, so a writer that collapses four
coordinates into one name is refused at hand-off with the three missing files
named.

**A file this store did not name cannot answer.** Every file in an entry is
read back through `naming.read_back`; a neighbour that does not parse, is not
listed, or whose digest has moved is refused by name rather than returned.

**The entry also states two things about its own validity.** A published entry carries the **fingerprint of every row
the invalidation matrix gives its kind** -- a `machine_spec`'s capacity and
tokenizer terms are invalidated by different things -- and the **state of
every gate that shaped it**. Both are required at publish rather than checked
at load: a campaign that runs for hours and
produces an entry nobody can certify has spent the hours, and the flag that
decided the gate is not around to be asked afterwards. `read` states an entry;
`load` states it *and certifies it*, and they are two calls because the first
is how a person inspects something the second has just refused.

The flag that downgrades a mismatch to a warning lives on `load`, and it
covers the invalidation matrix and nothing else -- a gate disagreement and a
comparison that could not be made have no warning form.
"""

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .fingerprints import Conditions, Fingerprint, OnMismatch, fingerprint, verify
from .gates import GateState
from .keys import Key, Kind, refuse_not_a_key
from .matrix import Row, rows_for
from .naming import AXES, RankCoords, Topology, member_name, read_back
from .provenance import Provenance
from .resolution import Resolution
from .rules import ArtifactRefusal, Rule

SCHEMA_VERSION = 2
ENTRY_FILE = "entry.json"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@contextlib.contextmanager
def _legible(directory: Path):
    """Read an entry document, turning a malformed one into a named refusal.

    A hand-edited `entry.json` otherwise surfaces as `JSONDecodeError`,
    `KeyError` or `ValueError` -- three tracebacks that say nothing about
    which artifact answered, which is the shape of failure this store is
    about. The module argues that the path is a place and never the
    authority; that is only true if a document the store does not recognise
    is declined by name.
    """
    try:
        yield
    except ArtifactRefusal:
        raise
    except (ValueError, KeyError, TypeError, AttributeError) as malformed:
        raise ArtifactRefusal(
            Rule.RESOLUTION,
            f"{directory / ENTRY_FILE} is not an entry document: "
            f"{type(malformed).__name__}: {malformed}",
            "an entry states its kind, key, topology, provenance, notes and "
            "members; this one was edited or written by something else, and "
            "a store that read it anyway would answer under a name it cannot "
            "support",
        ) from malformed


@dataclass(frozen=True, slots=True)
class Recorded:
    """An entry as its document names its files: its key as it restates it, its
    topology, its members' digests, its own digest, and where it is."""

    key: Key
    topology: Topology
    members: Mapping[str, str]
    digest: str
    directory: Path

    def read_member(self, stem: str, coords: RankCoords, extension: str) -> bytes:
        """One rank's file, named by the same function that wrote it."""
        if coords.topology != self.topology:
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"this entry was written at {self.topology.text} and the read "
                f"asks at {coords.topology.text}",
                "read at the topology the entry states; a read side that "
                "assumes a narrower run asks for a name no writer produced, "
                "which is how two workers died on a bare FileNotFoundError",
            )
        name = member_name(stem, coords, extension)
        if name not in self.members:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"`{name}` is not in {self.key}",
                "this entry holds " + ", ".join(sorted(self.members)),
            )
        payload = (self.directory / name).read_bytes()
        seen = _digest(payload)
        if seen != self.members[name]:
            raise ArtifactRefusal(
                Rule.IMMUTABLE,
                f"`{name}` digests {seen}, and this entry records "
                f"{self.members[name]}",
                "the file changed after it was published, so the entry that "
                "was reviewed under this digest no longer exists",
            )
        return payload


@dataclass(frozen=True, slots=True)
class Entry(Recorded):
    """One published artifact, as it was read back off the disk."""

    provenance: Provenance
    notes: str
    fingerprints: Mapping[Row, Fingerprint]
    gate_state: GateState


class ArtifactStore:
    """A root directory, and the one function from a key to a place in it."""

    def __init__(self, root: os.PathLike | str) -> None:
        self.root = Path(root)

    def directory_for(self, key: object) -> Path:
        """Where an entry lives. The only function that turns a key into a path."""
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        return self.root / key.kind.value / key.dirname()

    def key_from_token(self, kind: Kind, token: str) -> Key:
        """The key of the `kind` entry whose directory ends in `token`, the
        digest `Key.token` prints, so a command line can name an entry by it."""
        folder = self.root / kind.value
        found = sorted(
            place
            for place in (folder.iterdir() if folder.is_dir() else ())
            if place.name.endswith(token) and (place / ENTRY_FILE).is_file()
        )
        if len(found) != 1:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{len(found)} {kind} entries in {self.root} end in {token!r}"
                + "".join(f"; {place.name}" for place in found),
                f"name one {kind} entry by the token `Key.token` prints for its key",
            )
        with _legible(found[0]):
            document = json.loads((found[0] / ENTRY_FILE).read_bytes())
            key = Key.of(Kind(document["kind"]), **document["key"])
        if key.kind is not kind or key.token() != token:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{found[0]} ends in {token!r} and restates {key}, whose token "
                f"is {key.token()!r}",
                "an entry restates its key, so a directory moved or renamed by "
                "hand is found out rather than believed",
            )
        return key

    def publish(
        self,
        key: object,
        *,
        provenance: Provenance,
        topology: Topology,
        conditions: Conditions,
        gates: GateState,
        members: Mapping[str, bytes],
        notes: str = "",
    ) -> Entry:
        """Hand an entry off. Once per key, with every rank present.

        `conditions` and `gates` have no defaults on purpose. An entry that
        states neither is one a `load` can only decline, and it would decline
        it after the campaign that produced it had already run; and the flag
        that decided a gate is not available to be asked once the worker that
        read it has exited.
        """
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        if not isinstance(provenance, Provenance):
            raise ArtifactRefusal(
                Rule.PROVENANCE,
                f"{provenance!r} is not a provenance stanza",
                "name what produced this entry and out of which trees",
            )
        if not isinstance(topology, Topology):
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"{topology!r} is not a topology",
                "state every axis's width, so each member can name its rank",
            )
        if not isinstance(notes, str):
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"`notes` holds {type(notes).__name__}, not text",
                "the notes are inside the digested document, so anything that "
                "is not text would leave the entry's identity depending on how "
                "json.dumps happened to render it",
            )
        prints = self._fingerprints(key, conditions, gates)
        stated = key.values
        differ = [
            a
            for a in AXES
            if stated.get(a, getattr(topology, a)) != getattr(topology, a)
        ]
        if differ:
            raise ArtifactRefusal(
                Rule.ONE_NAMING_FUNCTION,
                f"{key} states {', '.join(f'{a}={stated[a]}' for a in differ)} "
                f"and its members are written at {topology.text}",
                "publish at the topology the key states; an entry whose ranks "
                "are not the layout it is keyed by answers for ranks it does "
                "not hold",
            )
        self._check_members(key, topology, members)
        if self.directory_for(key).exists():
            self._refuse_overwrite(key, members)
        self._write(
            self.directory_for(key),
            {
                "schema_version": SCHEMA_VERSION,
                "kind": key.kind.value,
                "key": key.values,
                "topology": topology.widths,
                "provenance": provenance.as_json(),
                "fingerprints": {
                    row.field: print_.as_json() for row, print_ in prints.items()
                },
                "gates": gates.as_json(),
                "notes": notes,
                "members": {
                    name: _digest(body) for name, body in sorted(members.items())
                },
            },
            members,
        )
        return self.read(key)

    def _fingerprints(
        self, key: Key, conditions: Conditions, gates: GateState
    ) -> Mapping[Row, Fingerprint]:
        """One fingerprint per row the matrix gives this kind, or a refusal."""
        if not isinstance(conditions, Conditions):
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"{conditions!r} is not a set of conditions",
                "state every axis; an entry records the "
                "fingerprint of its own dependency row and cannot take one "
                "from conditions nobody stated",
            )
        if not isinstance(gates, GateState):
            raise ArtifactRefusal(
                Rule.GATE_STATE,
                f"{gates!r} is not a gate state",
                "state the gates that shaped this entry, even when there are "
                "none; `GateState.of()` says that and an omission does not",
            )
        return {row: fingerprint(row, conditions) for row in rows_for(key.kind)}

    def read(self, key: object) -> Entry:
        """The entry at a key, checked against what it says about itself."""
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        directory = self.directory_for(key)
        if not (directory / ENTRY_FILE).is_file():
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"no entry for {key}{self._nearest(key)}",
                f"nothing is published at {directory}; a refusal names the key "
                "that missed, because a missing price is not a missing graph",
            )
        raw = (directory / ENTRY_FILE).read_bytes()
        with _legible(directory):
            document = json.loads(raw)
            if document.get("schema_version") != SCHEMA_VERSION:
                raise ArtifactRefusal(
                    Rule.RESOLUTION,
                    f"{directory} states schema_version "
                    f"{document.get('schema_version')!r}",
                    f"this reader understands version {SCHEMA_VERSION}",
                )
            stored = Key.of(Kind(document["kind"]), **document["key"])
            topology = Topology.from_mapping(document["topology"])
            provenance = Provenance.from_json(document["provenance"])
            wanted = {row.field: row for row in rows_for(stored.kind)}
            stale = sorted(set(document["fingerprints"]) - set(wanted))
            if stale:
                raise ArtifactRefusal(
                    Rule.INVALIDATED,
                    f"{directory} records a fingerprint for "
                    f"`{', '.join(stale)}`, which is not a row this kind is "
                    "checked against",
                    "the matrix moved under this entry; re-measure it, "
                    "because a recorded row nothing compares is a check "
                    "that silently stopped",
                )
            prints = {
                row: Fingerprint.from_json(document["fingerprints"][field])
                for field, row in wanted.items()
            }
            gate_state = GateState.from_json(document["gates"])
            notes, members = document["notes"], document["members"]
        if stored != key:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} holds {stored}, and was asked for {key}",
                "an entry restates its key, so a directory moved or renamed by "
                "hand is found out rather than believed",
            )
        self._check_directory(directory, topology, members)
        return Entry(
            key,
            topology,
            members,
            _digest(raw),
            directory,
            provenance,
            notes,
            prints,
            gate_state,
        )

    def read_recorded(self, kind: Kind, key: Mapping, digest: str) -> Recorded:
        """The `kind` entry a run recorded: the one whose document digests
        `digest` and restates `key`, compared as data rather than through
        `Key.of`, so an entry keyed by fields this tree's keys no longer have
        still answers. Only its key, topology and members are read, so nothing
        else it states is checked against this process. A publisher's staging
        directory is not an entry; a document that does not parse or states no
        key is passed over, and named when no entry answers."""
        folder, found, passed = self.root / kind.value, {}, []
        for path in sorted(folder.glob(f"[!.]*/{ENTRY_FILE}")):
            try:
                raw = path.read_bytes()
                if (document := json.loads(raw))["key"] == key:
                    found[_digest(raw)] = path.parent, document
            except (OSError, ValueError, KeyError, TypeError) as error:
                passed.append(
                    f"; {path} is not an entry document: "
                    f"{type(error).__name__}: {error}"
                )
        if digest not in found:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{folder} holds {len(found)} entries keyed {kind} "
                f"{json.dumps(dict(key), sort_keys=True)}, digesting "
                f"{sorted(found)}" + "".join(passed),
                f"the run recorded the one digesting {digest}",
            )
        directory, document = found[digest]
        with _legible(directory):
            topology = Topology.from_mapping(document["topology"])
            members = document["members"]
        return Recorded(
            Key(kind, tuple(key.items())), topology, members, digest, directory
        )

    def _nearest(self, key: Key) -> str:
        """The fields the nearest entry of the key's kind differs in, as text.

        A miss at `-tp 2 --enable-expert-parallel` beside an entry at `-tp 2`
        then says so, rather than leaving the reader to diff two keys by eye.
        Fields compare as their JSON text, so `True` and `1` differ.
        """
        best = None
        for path in sorted((self.root / key.kind.value).glob(f"*/{ENTRY_FILE}")):
            try:
                there = json.loads(path.read_bytes())["key"]
                differ = [
                    f"{name} ({there.get(name)!r} there)"
                    for name, value in key.fields
                    if json.dumps(there.get(name)) != json.dumps(value)
                ]
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if differ and (best is None or len(differ) < len(best)):
                best = differ
        return f"; the nearest entry differs in {', '.join(best)}" if best else ""

    def load(
        self,
        key: object,
        *,
        conditions: Conditions,
        gates: GateState,
        on_mismatch: OnMismatch = OnMismatch.REFUSE,
    ) -> Entry:
        """The entry at a key, certified against the conditions and gates in force.

        The gate state is checked first and the fingerprint second, because a
        measurement taken under other gates is not the thing the fingerprint
        describes -- the dead `PRICE_KERNELS` gate produced a price list whose
        every dependency read correctly and whose 164 per-kernel breakdowns
        were taken at a width the gate claimed to have excluded.

        `on_mismatch` is the explicit warning flag and reaches the matrix
        only. A
        gate disagreement refuses under either setting, and so does a reading
        that cannot be compared: there is no answer there to downgrade.
        """
        entry = self.read(key)
        entry.gate_state.check(gates)
        verify(entry.fingerprints, conditions, on_mismatch)
        return entry

    def answer(
        self,
        key: object,
        resolution: Resolution,
        *,
        conditions: Conditions,
        gates: GateState,
        on_mismatch: OnMismatch = OnMismatch.REFUSE,
    ) -> Entry | None:
        """Load an entry into a step's ledger: the entry, or a recorded miss.

        The refusal is recorded rather than raised so the step can name
        **every** key that missed instead of the first, which is the whole
        difference between `incomplete: N/2570` and a list. `Resolution.
        require_complete` is where the step refuses, and a caller that never
        calls it has a ledger and no gate -- so a caller that means "this must
        answer" should call `load`.
        """
        if not isinstance(key, Key):
            refuse_not_a_key(key)
        try:
            entry = self.load(
                key, conditions=conditions, gates=gates, on_mismatch=on_mismatch
            )
        except ArtifactRefusal as declined:
            resolution.missed(key, declined)
            return None
        resolution.answered(entry.key, entry.digest)
        return entry

    def _check_members(
        self, key: Key, topology: Topology, members: Mapping[str, bytes]
    ) -> None:
        if not isinstance(members, Mapping) or not members:
            raise ArtifactRefusal(
                Rule.EVERY_RANK_WRITES,
                f"{key} was handed off with no members",
                "an entry holds the files it was published for",
            )
        groups: dict[tuple[str, str], set[str]] = {}
        for name, body in members.items():
            if not isinstance(body, (bytes, bytearray)):
                raise ArtifactRefusal(
                    Rule.EVERY_RANK_WRITES,
                    f"`{name}` holds {type(body).__name__}, not bytes",
                    "publish the bytes that were written, so a digest is of "
                    "what a reader will get",
                )
            stem, coords, extension = read_back(name, topology)
            groups.setdefault((stem, extension), set()).add(coords.suffix)
        wanted = {rank.suffix for rank in topology.ranks()}
        for (stem, extension), seen in sorted(groups.items()):
            if seen != wanted:
                missing = ", ".join(
                    f"{stem}{suffix}.{extension}" for suffix in sorted(wanted - seen)
                )
                raise ArtifactRefusal(
                    Rule.EVERY_RANK_WRITES,
                    f"`{stem}.*.{extension}` covers {len(seen)} of "
                    f"{topology.rank_count} ranks of {topology.text}; missing {missing}",
                    "every rank writes its own file, and a single-writer path "
                    "is indistinguishable from a correct one by inspection",
                )

    def _refuse_overwrite(self, key: Key, members: Mapping[str, bytes]) -> None:
        try:
            standing = self.read(key)
        except ArtifactRefusal as unreadable:
            raise ArtifactRefusal(
                Rule.IMMUTABLE,
                f"{self.directory_for(key)} is in the way of {key} and is not "
                f"an entry this store can read: {unreadable.what}",
                "an occupied place is not a free one, whatever is in it; move "
                "or remove that directory deliberately, and do not let a "
                "publish decide it on a reader's behalf",
            ) from unreadable
        incoming = {name: _digest(body) for name, body in members.items()}
        same = incoming == dict(standing.members)
        raise ArtifactRefusal(
            Rule.IMMUTABLE,
            f"{key} was handed off at {standing.digest} and this publish holds "
            + ("the same members" if same else "different members"),
            "publish a new entry under a key that states what differs; a "
            "regeneration that changes only the notes still destroys the "
            "digest the entry was reviewed as",
        )

    def _check_directory(
        self, directory: Path, topology: Topology, members: Mapping[str, str]
    ) -> None:
        present = {item.name for item in directory.iterdir()} - {ENTRY_FILE}
        for name in sorted(present):
            read_back(name, topology)
        unlisted = sorted(present - set(members))
        if unlisted:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} holds {', '.join(unlisted)}, which it does not list",
                "an entry lists every file it answers from; a neighbour nobody "
                "recorded is what a read that dropped its rank coordinates was "
                "about to answer from",
            )
        absent = sorted(set(members) - present)
        if absent:
            raise ArtifactRefusal(
                Rule.RESOLUTION,
                f"{directory} lists {', '.join(absent)}, which is not there",
                "the entry is incomplete; republish it under a new key",
            )

    def _write(
        self, destination: Path, document: dict, members: Mapping[str, bytes]
    ) -> None:
        """Build the entry beside its place, then move it there in one step.

        The `except` catches more than `OSError` because it did not, and the
        staging directory survived an exception raised on the way to the
        rename (#169). A `TypeError` out of `json.dumps` left
        `.m-2-8f37148e79f6.sx267hib` behind and reported nothing about it --
        litter that is never reclaimed, from a failure that never published.
        The narrow `except` was right about the failure it was written for and
        silent about every other one.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}.")
        )
        try:
            for name, body in members.items():
                (staging / name).write_bytes(body)
            (staging / ENTRY_FILE).write_bytes(
                json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
            )
            for item in staging.iterdir():
                item.chmod(0o444)
            os.rename(staging, destination)
        except (OSError, TypeError, ValueError) as clash:
            shutil.rmtree(staging, ignore_errors=True)
            raise ArtifactRefusal(
                Rule.IMMUTABLE,
                f"{destination} could not be created: {type(clash).__name__}: {clash}",
                "an entry is moved into place onto a name a published entry "
                "already occupies, so a second publisher loses the race rather "
                "than the entry; a publish that fails on the way there takes "
                "its half-built staging directory with it",
            ) from clash

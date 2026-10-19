# SPDX-License-Identifier: MIT
"""The parity workload: distinct completions sent on a schedule, to a real or a simulated run.

    python -m atom.compass.parity.traffic PORT MODEL RECORD [REQUESTS [GAP_S [PREFIX_WORDS [IMAGES [ALONE [LONG [FAMILIES]]]]]]]

Request k's prompt starts with PREFIX_WORDS words every request shares (none
by default), then k; with FAMILIES above 1 it shares them only with family
k % FAMILIES, whose words start with its number. With IMAGES it is a chat
message after that many images (`image`). It asks for a fixed number of tokens with
``ignore_eos``, streamed; request 0 for LONG tokens when LONG is above 0. On a real run (`ATOM_COMPASS_RUN` unset)
request k is sent GAP_S * k wall seconds after the first and not before the
server has admitted request k - 1, each on its own thread; with ALONE 1,
request 0 is first sent alone and read to its end, and request k is sent
GAP_S * (k - 1) seconds after request 1. The worker clock's
reading as each was sent, and as its response headers arrived (the server
answers once it has handed the request to the engine), goes to `ARRIVALS` in
RECORD, the real run's parity record directory. On a simulated run this
process is the traffic LP: it sends request k at its send reading less the
first one, on the run's clock, and takes every stream event the authority
releases until the finish.
"""

import base64
import http.client
import io
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.carriers import sse_stamp, tracestate_with
from atom.compass.clock_transport import connect
from atom.compass.parity import ARRIVALS
from atom.utils.clock import LPRuntime

DONE = "data: [DONE]"
WALL_S = 3600.0
#: The prose prompts are cut from, written for this file, one paragraph a line,
#: long enough that no default-sized prompt wraps around it. A prompt that
#: repeats itself is continued by copying, so a speculative decoder's every
#: draft is accepted.
TEXT = """
The town sat where the river bent twice before it reached the sea, and the people who lived there liked to say that the water could not make up its mind. In the spring it ran high and brown with snow from the hills, and the old stone bridge would hum when the current pushed against its piers. Children were told to stay off the banks in those weeks, though most of them went down anyway to watch branches and whole trees turn slowly in the flood. By early summer the river had calmed into a wide green road, and the ferry that crossed below the mill began its season again. The ferry was a flat boat on a steel cable, pulled across by a small engine that coughed every morning before it agreed to start. Its owner, a tall woman named Ruth, had run it for nearly thirty years and claimed she could tell the depth of the water by the sound of the engine alone.
On the north side of the river stood the market square, paved with uneven stones that had been laid by hand long before anyone could remember. Every Saturday the square filled with tables and carts. Farmers brought eggs, cheese, onions, and jars of honey, while a man from the coast sold dried fish wrapped in paper and told stories that grew longer every year. There was a baker who arrived before dawn with loaves still warm in wooden crates, and a pair of sisters who mended shoes, umbrellas, and anything else people carried to them. The noise of the market rose through the morning and then faded after noon, when the sun reached the middle of the square and the shade under the chestnut trees became the only comfortable place to stand. By the middle of the afternoon the stones were swept, the tables folded, and the pigeons had taken back what they considered their own.
The school was a long building of red brick with tall windows that let in more cold than light in winter. It had been built for twice as many pupils as it now held, so several rooms at the far end were used for storage, and the children believed that one of them was haunted by a teacher who had never finished marking her papers. The head teacher, Mr. Okafor, did nothing to discourage the story. He thought a school ought to have at least one mystery, and he enjoyed the way the youngest pupils lowered their voices when they walked past the closed doors. In the main rooms the desks were old but solid, carved over the decades with initials, small boats, and a remarkably accurate map of the river that no one had ever claimed.
Most of the work in the town had once come from the mill, which ground grain from the valley farms and later made paper for a printing company in the city. The mill closed when the company moved its presses abroad, and for a few years the building stood empty, its wheel locked in place by rust and weeds. Then a group of residents bought it for very little money and began to repair it one room at a time. Now the ground floor held a workshop where people could borrow tools, the second floor had become a small library, and the top floor, with its view of the bridge and the hills beyond, was rented out for weddings, meetings, and the occasional concert. The wheel had been freed and turned again, though it powered nothing more than a set of lights along the path.
Summer brought visitors who came by train and walked down the hill from the station carrying bags that were usually too heavy. They swam in the shallow pool below the weir, ate lunch on the grass, and bought postcards of the bridge from the shop next to the post office. The people of the town had mixed feelings about them. The visitors spent money, which was welcome, but they also left gates open, asked the same questions every day, and sometimes tried to feed the swans, which the swans accepted with very poor grace. Ruth on the ferry was patient with them, mostly. She answered every question about the river as if it had never been asked before, and only once, after a very long afternoon, told a young man that the water was bottomless and that the fish at the far end of it could speak French.
In late summer the fields around the town turned gold, and the harvest pulled everyone into it whether they farmed or not. Teachers, clerks, and shopkeepers took a week off to help bring in the wheat, and the evenings ended with long tables set out in the yard behind the mill. The meals were simple: bread, soup, roasted vegetables, and whatever fruit was ripe. Someone always brought a fiddle, and someone else always complained that the fiddle was out of tune, and both of these things were considered part of the tradition. When the last wagon came in, the oldest farmer in the valley tied a small bundle of wheat to the gate of his barn, where it stayed until the following spring as a quiet thank you to the land.
Autumn came in slowly, first as a change in the color of the light and then as a chill that waited in the shadows until midday. The chestnut trees in the square dropped their nuts onto the stones, and children filled their pockets with them for games that seemed to have no fixed rules. The river began to rise again with the rain from the hills, and the ferry made its final crossings of the year with fewer passengers and more cargo: sacks of apples, crates of potatoes, a piano once, which Ruth refused to discuss afterward. Smoke rose from the chimneys in the evening, and the smell of burning wood hung over the lanes as people walked home in the early dark, their breath showing in the cold air.
The library on the second floor of the mill was busiest in the colder months. It held about four thousand books, nearly all of them donated, and they were arranged according to a system invented by the volunteer who had sorted them. Novels were shelved by the color of their covers, history by the century it described, and books about gardening sat next to books about cooking because, as the volunteer explained, one naturally led to the other. Visitors from outside found the system confusing, but the regular readers learned it quickly and defended it with surprising passion. On Tuesday evenings a reading group met by the window, and on Thursday afternoons an old sailor taught anyone who wanted to learn how to tie knots, using a length of rope he kept in a canvas bag.
Winter in the town was quiet but rarely dull. When the snow came it settled thickly on the roofs and turned the hills into a clean white wall behind the houses. The council kept one road open to the station with a plow that was older than most of the people who drove it, and the rest of the lanes were cleared by hand, each household responsible for the stretch in front of its own door. Some years the river froze at the edges, though never all the way across, and the ducks gathered in the open water near the bridge where the current was strongest. The children built forts in the school yard and fought long battles that ended only when someone was called home for dinner, at which point both sides declared victory and agreed to continue the next day.
Near the end of the year the town held a festival of lights. Every house put a lantern in its front window, and on the longest night people walked in a slow line from the square to the bridge carrying candles in glass jars. Nobody was quite sure when the custom had started. One story said it began after a winter flood, when the lanterns were lit to guide a lost boat back to shore. Another said that a merchant had simply wanted to sell more candles and had succeeded beyond his wildest hopes. Mr. Okafor told his pupils both stories and then asked them to write a third one of their own, and the best of these were read aloud at the festival from the steps of the mill, where the wind did its best to carry the words away.
Like every small place, the town had its arguments. There was a long dispute about whether the bridge should be widened for larger trucks, which divided neighbors and filled the letters page of the weekly paper for most of a year. In the end the bridge stayed as it was, partly because the engineers found that its old stones were stronger than expected, and partly because nobody could agree on who would pay. There was another argument about the clock on the church tower, which ran four minutes slow. Some people wanted it fixed, while others felt the delay gave the whole town a gentle sort of patience. The clock was eventually repaired, and within a month it had drifted back to exactly four minutes behind, which many took as a sign that it knew best.
The weekly paper itself was produced by two people in a narrow office above the bakery. The editor wrote most of the articles, took most of the photographs, and sold most of the advertisements, while his partner handled the printing, the delivery, and the steady stream of complaints. The paper reported on council meetings, school plays, lost dogs, and the results of the vegetable show, where the competition for the heaviest marrow had once become so fierce that the judges asked for a police officer to stand beside the scales. It rarely carried news from beyond the valley, and readers seemed to like it that way. When something important happened in the wider world, they said, they would hear about it soon enough, and in the meantime there were marrows to weigh.
Spring returned the way it always did, with mud first and flowers second. The swallows came back to the eaves of the mill, and the farmers walked their fields to see what the frost had left them. Ruth spent two weeks on the bank repairing the ferry, painting its rails, checking the cable for wear, and persuading the engine to cooperate. On the first warm morning she took it across empty, just to listen to it, and the people waiting on the far side said later that she had been smiling the whole way. The river was still high and fast, but it had lost the angry brown of the thaw and begun to clear, and on a calm day you could see the stones of the riverbed shining through the water near the shore.
Over the years some of the young people left for the cities, as young people from small towns always have, and some of them came back. Those who returned often said they had not realized how much they missed the sound of the river until they had lived somewhere it could not be heard. A few of them brought new ideas with them: a cafe in the old station house, a small workshop that built furniture from fallen trees, and a website that sold the honey from the valley to customers in other countries. The older residents watched these changes with a mixture of suspicion and pride. They grumbled about the prices at the cafe and then went there every morning, and they told visitors that the honey had always been the best in the region, long before anyone thought to put it on a screen.
Ruth retired from the ferry in the end, though it took her three attempts. The first time she announced it, she changed her mind within a week because the young man who was meant to take over could not start the engine. The second time she lasted a full month before deciding that the new owner was crossing too fast and frightening the swans. The third time she stayed retired, mostly, and could be found most afternoons on a bench by the landing, offering advice to the new ferry keeper whether it was wanted or not. He took it well. He said that learning the river from Ruth was like learning a language from someone who had invented half of the words, and that he expected to be a student for many years to come.
In the evenings, when the light went soft and the swallows dropped low over the water, people still walked down to the bridge to lean on the stone wall and look upstream toward the hills. They talked about the weather, the harvest, the price of bread, and the things their grandparents had told them about the same view. Some of them stayed until the stars came out over the valley. The river went on bending twice before it reached the sea, carrying leaves and light and the occasional lost hat, and if it had made up its mind about anything at all, it was keeping that decision to itself.
"""
WORDS = TEXT.split()


def image(k: int) -> str:
    """Request k's image: a 224 x 224 PNG of one colour, as a data URL."""
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (224, 224), (37 * k % 256, 91 * k % 256, 53 * k % 256)).save(
        out, "PNG"
    )
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def bodies(
    model: str,
    n: int,
    prefix_words: int = 0,
    images: int = 0,
    long: int = 0,
    families: int = 1,
) -> list[dict]:
    """Request k: `prefix_words` words every request shares, then about
    48 + 31k words starting with k; 16 to 96 output tokens, or `long` for
    request 0 when `long` is above 0. With `images`, a chat message whose text
    follows that many copies of `image(k)`. With `families` above 1, request k
    shares the words only with family k % `families`, whose words start with
    its number."""
    prefix = "".join(f"{WORDS[i % len(WORDS)]} " for i in range(prefix_words))
    out = []
    for k in range(n):
        body = {
            "model": model,
            "prompt": (f"{k % families} " if families > 1 else "")
            + f"{prefix}{k} "
            + " ".join(WORDS[(k + i) % len(WORDS)] for i in range(48 + 31 * k)),
            "max_tokens": long if long > 0 and k == 0 else 16 + 16 * (k % 6),
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
        }
        if images:
            part = {"type": "image_url", "image_url": {"url": image(k)}}
            text = {"type": "text", "text": body.pop("prompt")}
            body["messages"] = [{"role": "user", "content": [part] * images + [text]}]
        out.append(body)
    return out


def post(port: int, body: dict, headers: dict):
    """The streamed response's lines, once the server has admitted the request
    (its headers follow the engine's `add_request`); raises unless it answers 200."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WALL_S)
    headers = {"content-type": "application/json", **headers}
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    conn.request("POST", path, json.dumps(body), headers)
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    return (raw.decode().rstrip("\r\n") for raw in response)


def real(
    port: int,
    requests: list[dict],
    gap_s: float,
    record: Path,
    wait: bool = True,
    prefix_words: int = 0,
    images: int = 0,
    warmup: bool = False,
    long: int = 0,
    families: int = 1,
) -> None:
    """Send request k at `gap_s` * k, once k - 1 is admitted when `wait`, read
    each to its end, write `ARRIVALS` with the requests' `prefix_words`,
    `images`, `long` and `families`. With `warmup`, request 0 is read to its end alone and the gaps
    count from then, so the rest arrive into running batches rather than all
    during the first forward.
    Through a prefill-decode router a request is admitted only once its
    prefill returns, so `wait` keeps prefills apart."""
    sent, start = [0] * len(requests), time.monotonic()
    admitted = [threading.Event() for _ in requests]
    admitted_ns = [0] * len(requests)

    def one(k: int) -> None:
        time.sleep(max(0.0, start + k * gap_s - time.monotonic()))
        try:
            if k and wait:
                admitted[k - 1].wait()
            sent[k] = time.monotonic_ns()
            lines = post(port, requests[k], {})
            admitted_ns[k] = time.monotonic_ns()
        finally:
            admitted[k].set()
        if DONE not in lines:
            raise RuntimeError(f"request {k} ended before {DONE}")

    first = 0
    if warmup:
        one(0)
        first, start = 1, time.monotonic() - gap_s
    with ThreadPoolExecutor(len(requests)) as pool:
        list(pool.map(one, range(first, len(requests))))
    arrivals = {
        "sent_ns": sent,
        "admitted_ns": admitted_ns,
        "prefix_words": prefix_words,
        "images": images,
        "long": long,
        "families": families,
    }
    (record / ARRIVALS).write_text(json.dumps(arrivals))


class Traffic:
    """The traffic LP of a simulated run: request k at `offsets[k]` on its clock."""

    def __init__(self, run: dict, port: int, requests: list, offsets: list) -> None:
        table = compass_run.channel_table(run)
        (http,) = table.channels_from(compass_run.TRAFFIC)
        (stream,) = table.channels_into(compass_run.TRAFFIC)
        self.http, self.stream, self.port = http.name, stream.name, port
        conn = connect(compass_run.TRAFFIC, run["clock_endpoint"])
        self.rt = LPRuntime(compass_run.TRAFFIC, table, conn)
        self.requests, self.offsets = requests, offsets
        self.cv = threading.Condition()
        self.frames: dict[int, str] = {}  # stream seq -> its data line, read, not taken
        self.errors: list[BaseException] = []
        self.done = 0

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for body, at in sorted(zip(self.requests, self.offsets), key=lambda r: r[1]):
            while rt.now < at:
                rt.next_event(at)
                self._take()
            stamp = tracestate_with(None, *rt.stamp_send(self.http))
            threading.Thread(target=self._read, args=(body, stamp), daemon=True).start()
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()
        if self.done != len(self.requests):
            raise RuntimeError(f"{len(self.requests) - self.done} requests never ended")

    def _read(self, body: dict, tracestate: str) -> None:
        try:
            stamp = None
            for line in post(self.port, body, {"tracestate": tracestate}):
                stamp = sse_stamp(line) or stamp
                if line.startswith("data:") and stamp is not None:
                    self.rt.check_arrival(self.stream, *stamp)
                    with self.cv:
                        self.frames[stamp[1]] = line
                        self.cv.notify_all()
                    stamp = None
        except BaseException as e:  # noqa: BLE001 - the clock owner raises it
            with self.cv:
                self.errors.append(e)
                self.cv.notify_all()

    def _take(self) -> None:
        """Take every released stream event, waiting for any not read yet."""
        with self.rt.lock:
            due = sorted(self.rt.released[self.stream] - self.rt.handled[self.stream])
        for seq in due:
            with self.cv:
                if not self.cv.wait_for(
                    lambda seq=seq: seq in self.frames or self.errors, WALL_S
                ):
                    raise RuntimeError(f"stream seq {seq} released and never read")
                if self.errors:
                    raise self.errors[0]
                self.done += self.frames.pop(seq) == DONE
            with self.rt.lock:
                self.rt.count_done_locked(self.stream, seq)


if __name__ == "__main__":
    port, model, record, *rest = sys.argv[1:]
    run, record = compass_run.spec(), Path(record)
    if run is None:
        n, gap_s = int(rest[0]) if rest else 64, float(rest[1]) if rest[1:] else 0.05
        prefix, images, alone, long, families = (
            int(rest[i]) if rest[i:] else 0 for i in (2, 3, 4, 5, 6)
        )
        requests = bodies(model, n, prefix, images, long, families)
        real(
            int(port),
            requests,
            gap_s,
            record,
            prefix_words=prefix,
            images=images,
            warmup=alone == 1,
            long=long,
            families=families,
        )
    else:
        arrivals = json.loads((record / ARRIVALS).read_text())
        sent, prefix = arrivals["sent_ns"], arrivals.get("prefix_words", 0)
        images, long = arrivals.get("images", 0), arrivals.get("long", 0)
        families = arrivals.get("families", 1)
        requests = bodies(model, len(sent), prefix, images, long, families)
        offsets = [(t - min(sent)) / 1e9 for t in sent]
        Traffic(run, int(port), requests, offsets).run()

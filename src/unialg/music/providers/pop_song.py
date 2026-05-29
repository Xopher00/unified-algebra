"""Render a categorical study of a musical phrase and its transformations.

The DSL generates recursive pitch phrases through coalgebras and `ana`, then
applies whole-phrase musical transformations through `cata`.  Python remains
the MIDI scheduling boundary for this provider.
"""

from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from unialg.main import compile_program


ALL_ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
DSL_PARSE_RECURSION_LIMIT = 10_000


@dataclass(frozen=True)
class Performance:
    """Temporary MIDI-boundary annotation for one DSL-produced section."""

    label: str
    beats: tuple[float, ...]
    octave: int
    velocity: int


@dataclass(frozen=True)
class RenderedEvent:
    chord: list[str]
    melody: str
    beats: float
    octave: int
    velocity: int


# This mirrors the order of `song` below to schedule the DSL-produced pitch
# phrases at the MIDI output boundary.
PERFORMANCE_SCORE = (
    Performance("intro", (2.0, 1.0, 2.0), 4, 44),
    Performance("verse", (1.5, 0.5, 1.0, 1.0, 1.5, 0.5, 1.5), 4, 62),
    Performance("answer", (0.75, 0.75, 1.0, 1.5, 0.75, 0.75, 2.0), 4, 68),
    Performance("chorus", (0.75, 0.5, 0.75, 1.0, 0.75, 0.5, 1.75), 5, 86),
    Performance("return", (1.0, 0.5, 1.0, 1.0, 1.0, 0.5, 2.0), 4, 66),
    Performance("bridge", (1.5, 0.75, 0.75, 1.5, 2.0), 4, 54),
    Performance("final chorus", (0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 2.0), 5, 94),
    Performance("outro", (1.5, 1.0, 3.0), 4, 42),
)


def _song_source() -> str:
    """Return the musical object-and-transformation diagram in the DSL."""

    return """
load mingus

# RawEvent preserves harmonic quality as a coproduct branch:
# left = major (root x melody), right = minor (root x melody). Transformations
# act before chord spelling, so a shifted major triad remains a spelled major
# triad rather than a transposed list of already-rendered note names.
shape Tones = const[STRING] & const[STRING]
shape RawEventF = Tones | Tones
shape RawPhraseF = 1 | (RawEventF & x)
shape RawPhrase = fix RawPhraseF

# Binary musical development tree: a leaf or an event with two continuations.
# This is the executable binary specialization of a Tree-like branching form.
shape RawBranchF = 1 | (RawEventF & (x & x))
shape RawBranch = fix RawBranchF

shape EventF = const[List[STRING]] & const[STRING]
shape PhraseF = 1 | (EventF & x)
shape Phrase = fix PhraseF

# Tonal positions and raw musical events available to phrase coalgebras.
let d0 = reduce_accidentals
let d2 = interval_from_distance(id, '2')
let d4 = interval_from_distance(id, '4')
let d5 = interval_from_distance(id, '5')
let d7 = interval_from_distance(id, '7')
let d9 = interval_from_distance(id, '9')
let d11 = interval_from_distance(id, '11')
let maj(root, voice) = (root & voice) >> |0
let min(root, voice) = (root & voice) >> |1

let a = maj(d0, d4)
let b = maj(d5, d7)
let c = min(d9, d9)
let d = maj(d7, d11)
let e = min(d2, d4)
let f = maj(d7, d2)
let g = maj(d0, d0)

# A bounded section plan has seven optional slots. A live slot emits an event
# and advances; an empty slot terminates. One coalgebra unfolds every section.
let live(event, next_state) = (event & next_state) >> |1
let empty = id >> |0
let later(p) = p >> |1

let pos0 = |0
let pos1 = later(pos0)
let pos2 = later(pos1)
let pos3 = later(pos2)
let pos4 = later(pos3)
let pos5 = later(pos4)
let pos6 = later(pos5)
let end = later(later(later(later(later(later(later(delete)))))))

let unfold(s0, s1, s2, s3, s4, s5, s6) =
    pos0 >> ana[RawPhrase](
        s0 | (s1 | (s2 | (s3 | (s4 | (s5 | (s6 | empty)))))))

# Musical arrows on raw events and phrases.
let move_2 = interval_from_distance(id, '2')
let move_5 = interval_from_distance(id, '5')
let move_7 = interval_from_distance(id, '7')
let translate(step) = (step || step) && (step || step)
let map_raw(t) =
    cata[RawPhrase](
        (|0 >> roll[RawPhrase])
        | ((t || id) >> (|1 >> roll[RawPhrase])))
let shift(step) = map_raw(translate(step))
let swap_quality = (|1) | (|0)
let modal_shadow = map_raw(swap_quality)

# A branch is a space of alternative developments; selection folds one
# coherent route back into the sequential RawPhrase carrier for performance.
let branch_leaf = delete >> |0 >> roll[RawBranch]
let fork(event, left, right) =
    (event & (left & right)) >> |1 >> roll[RawBranch]
let continue(event, rest) = fork(event, rest, rest)
let choose_left =
    cata[RawBranch](
        (delete >> |0 >> roll[RawPhrase])
        | (([0] & ([1] >> [0])) >> (|1 >> roll[RawPhrase])))
let choose_right =
    cata[RawBranch](
        (delete >> |0 >> roll[RawPhrase])
        | (([0] & ([1] >> [1])) >> (|1 >> roll[RawPhrase])))

# Rendering is an interpretation of raw harmonic meaning into Mingus chord
# spelling. Inversions occur only after that interpretation.
let render_event = (major_triad || id) | (minor_triad || id)
let sound =
    cata[RawPhrase](
        (|0 >> roll[Phrase])
        | ((render_event || id) >> (|1 >> roll[Phrase])))
let turn = rotate_chord_inversion || id
let map_phrase(t) =
    cata[Phrase](
        (|0 >> roll[Phrase])
        | ((t || id) >> (|1 >> roll[Phrase])))
let turn_phrase = map_phrase(turn)

# Sections fill a bounded plan with live or empty slots, then transform and
# interpret the generated raw phrase.
let intro_material = unfold(
    live(a, pos1), live(b, pos2), live(c, pos3),
    empty, empty, empty, empty)
let intro = intro_material >> sound >> turn_phrase
let seed_raw = unfold(
    live(a, pos1), live(b, pos2), live(c, pos3), live(d, pos4),
    live(e, pos5), live(f, pos6), live(g, end))
let seed = seed_raw >> sound
let answer_material = unfold(
    live(g, pos1), live(f, pos2), live(e, pos3), live(d, pos4),
    live(c, pos5), live(b, pos6), live(a, end))
let answer = answer_material >> shift(move_2) >> sound

# Chorus development branches after a common opening. The two cata policies
# produce different seven-event paths from one branching musical object.
let bright4 = continue(g, branch_leaf)
let bright3 = continue(f, bright4)
let bright2 = continue(e, bright3)
let bright1 = continue(d, bright2)
let shadow4 = continue(a, branch_leaf)
let shadow3 = continue(d, shadow4)
let shadow2 = continue(g, shadow3)
let shadow1 = continue(e, shadow2)
let chorus_split = fork(c, bright1, shadow1)
let chorus_development = continue(a, continue(b, chorus_split))
let chorus_path = chorus_development >> choose_left
let final_path = chorus_development >> choose_right
let chorus = chorus_path >> shift(move_5) >> sound >> turn_phrase
let verse_return = seed_raw >> sound >> turn_phrase
let bridge_material = unfold(
    live(c, pos1), live(e, pos2), live(g, pos3), live(d, pos4),
    live(a, pos5), empty, empty)
let bridge = bridge_material >> modal_shadow >> shift(move_7) >> sound >> turn_phrase
let final_chorus = final_path >> shift(move_5) >> sound >> turn_phrase >> turn_phrase
let outro_material = unfold(
    live(c, pos1), live(b, pos2), live(g, pos3),
    empty, empty, empty, empty)
let outro = outro_material >> sound >> turn_phrase

let song = intro & seed & answer & chorus & verse_return & bridge & final_chorus & outro
"""


def _phrase_events(phrase):
    """Walk a runtime Phrase value, yielding (chord, melody) pairs."""

    while phrase[0] == "right":
        event_data, phrase = phrase[1]
        yield event_data


def _unpack_phrases(value, n: int) -> list:
    """Unpack a left-nested `&` product into its section phrase values."""

    parts = []
    for _ in range(n - 1):
        value, right = value
        parts.append(right)
    parts.append(value)
    parts.reverse()
    return parts


def to_midi(name: str, octave: int) -> int | None:
    if not name or name[0].upper() not in NOTE:
        return None
    pitch = NOTE[name[0].upper()] + (octave + 1) * 12
    for accidental in name[1:]:
        if accidental == "#":
            pitch += 1
        elif accidental == "b":
            pitch -= 1
    return max(21, min(108, pitch))


def voiced_chord_pitches(chord: Iterable[str], octave: int) -> list[int]:
    """Render a chord spelling from bass upward, retaining DSL inversions."""

    pitches: list[int] = []
    for note in chord:
        pitch = to_midi(note, octave)
        if pitch is None:
            continue
        while pitches and pitch <= pitches[-1]:
            pitch += 12
        pitches.append(min(108, pitch))
    return pitches


def _mido():
    try:
        return importlib.import_module("mido")
    except ModuleNotFoundError as exc:
        raise RuntimeError("MIDI rendering requires the optional 'mido' package") from exc


def write_song(song_number: int, tonic: str, program, output_dir: Path) -> Path:
    mido = _mido()
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    harmony_track = mido.MidiTrack()
    melody_track = mido.MidiTrack()
    mid.tracks.extend((harmony_track, melody_track))
    harmony_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(94), time=0))
    harmony_track.append(mido.MetaMessage("track_name", name=f"Harmony: {tonic}", time=0))
    melody_track.append(mido.MetaMessage("track_name", name="Categorical transformations", time=0))

    phrases = _unpack_phrases(program.run(tonic), len(PERFORMANCE_SCORE))
    for performance, phrase in zip(PERFORMANCE_SCORE, phrases):
        events = list(_phrase_events(phrase))
        if len(events) != len(performance.beats):
            raise ValueError(
                f"performance annotation for {performance.label!r} has "
                f"{len(performance.beats)} durations for {len(events)} DSL events"
            )
        harmony_track.append(mido.MetaMessage("marker", text=performance.label, time=0))
        for (chord, melody), beats in zip(events, performance.beats):
            event = RenderedEvent(
                chord,
                melody,
                beats,
                performance.octave,
                performance.velocity,
            )
            duration = round(event.beats * mid.ticks_per_beat)
            chord_pitches = voiced_chord_pitches(event.chord, event.octave)
            melody_pitch = to_midi(event.melody, event.octave + 1)
            for pitch in chord_pitches:
                harmony_track.append(
                    mido.Message("note_on", note=pitch, velocity=event.velocity, time=0)
                )
            for i, pitch in enumerate(chord_pitches):
                harmony_track.append(
                    mido.Message("note_off", note=pitch, velocity=0, time=duration if i == 0 else 0)
                )
            if melody_pitch is not None:
                melody_track.append(
                    mido.Message("note_on", note=melody_pitch, velocity=event.velocity + 8, time=0)
                )
                melody_track.append(
                    mido.Message("note_off", note=melody_pitch, velocity=0, time=duration)
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"song_{song_number}_{tonic}.mid"
    mid.save(output)
    return output


def describe(source: str) -> None:
    print(source)
    print("# MIDI scheduling payloads attached at the Python output boundary:")
    for performance in PERFORMANCE_SCORE:
        print(
            f"# {performance.label:13s}: beats={performance.beats}, "
            f"octave={performance.octave}, velocity={performance.velocity}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3, help="number of MIDI studies to render")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--print-dsl",
        action="store_true",
        help="print DSL source and boundary performance annotations without rendering",
    )
    args = parser.parse_args()

    source = _song_source()
    if args.print_dsl:
        describe(source)
        return

    if sys.getrecursionlimit() < DSL_PARSE_RECURSION_LIMIT:
        sys.setrecursionlimit(DSL_PARSE_RECURSION_LIMIT)
    program = compile_program(source, target="song")
    rng = random.Random(args.seed)
    for number in range(1, args.count + 1):
        tonic = rng.choice(ALL_ROOTS)
        output = write_song(number, tonic, program, args.output_dir)
        print(f"song {number}: tonic={tonic} -> {output}")


if __name__ == "__main__":
    main()

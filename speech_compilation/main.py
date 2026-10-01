import csv
import glob
import os
import shutil

import gradio as gr

DATA_DIR = "data"
OUTPUT_DIR = "output"

LANGUAGE_NAMES = {
    "spa": "Spanish",
    "quy": "Quechua",
    "eng": "English",
}

FLASHCARDS_TEXT_LINES=2

def list_corpora():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "flashcards_*.csv")))
    return paths


def flashcards_label(path):
    return os.path.splitext(os.path.basename(path))[0]


def parse_flashcards(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        rows = list(reader)

    header = [h.strip() for h in rows[0]]
    lang1_col, lang2_col = header[1], header[2]

    entries = []
    for row in rows[1:]:
        if not row:
            continue
        entry_id, text1, text2 = row[0].strip(), row[1].strip(), row[2].strip()
        entries.append({"id": entry_id, "text1": text1, "text2": text2})

    return lang1_col, lang2_col, entries


def flashcards_path(flashcards_dir):
    return os.path.join(flashcards_dir, "flashcards.csv")


def audio_path(flashcards_dir, entry_id, lang_col):
    return os.path.join(flashcards_dir, "audio", f"{entry_id}_{lang_col}.wav")


def load_completed_ids(flashcards_dir):
    path = flashcards_path(flashcards_dir)
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["id"] for row in reader}


def append_flashcard(flashcards_dir, lang1_col, lang2_col, entry, audio1_rel, audio2_rel):
    path = flashcards_path(flashcards_dir)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["id", lang1_col, lang2_col, f"audio_{lang1_col}", f"audio_{lang2_col}"])
        writer.writerow([entry["id"], entry["text1"], entry["text2"], audio1_rel, audio2_rel])


def find_start_index(entries, completed_ids):
    for i, entry in enumerate(entries):
        if entry["id"] not in completed_ids:
            return i
    return len(entries)


def lang_prompt_label(lang_col):
    name = LANGUAGE_NAMES.get(lang_col.lower(), lang_col)
    return f"{name} ({lang_col})"


CUSTOM_CSS = """
.gradio-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}
.prompt-text textarea {
    font-size: 3.5rem !important;
    line-height: 1.4 !important;
}
"""

with gr.Blocks(title="Speech Compilation") as demo:
    flashcards_paths = list_corpora()

    state = gr.State(
        {
            "flashcards_path": None,
            "flashcards_dir": None,
            "lang1_col": None,
            "lang2_col": None,
            "entries": [],
            "index": 0,
        }
    )

    gr.Markdown("# Speech Compilation")

    with gr.Row():
        flashcards_dropdown = gr.Dropdown(
            choices=[(flashcards_label(p), p) for p in flashcards_paths],
            value=flashcards_paths[0] if flashcards_paths else None,
            label="Flashcards",
        )
        load_btn = gr.Button("Load flashcards")

    progress_md = gr.Markdown()

    with gr.Row():
        with gr.Column():
            lang1_label = gr.Markdown()
            text1 = gr.Textbox(label="Prompt text", lines=FLASHCARDS_TEXT_LINES, elem_classes=["prompt-text"])
            audio1 = gr.Audio(sources=["microphone"], type="filepath", label="Record")
            replay1 = gr.Audio(label="Playback", interactive=False)

        with gr.Column():
            lang2_label = gr.Markdown()
            text2 = gr.Textbox(label="Prompt text", lines=FLASHCARDS_TEXT_LINES, elem_classes=["prompt-text"])
            audio2 = gr.Audio(sources=["microphone"], type="filepath", label="Record")
            replay2 = gr.Audio(label="Playback", interactive=False)

    with gr.Row():
        save_btn = gr.Button("Save and continue", variant="primary")
        skip_btn = gr.Button("Skip entry")

    status_md = gr.Markdown()

    def do_load(flashcards_path, current_state):
        if not flashcards_path:
            return (
                current_state,
                "No flashcards selected.",
                "", "", "", "", None, None, None, None,
            )

        lang1_col, lang2_col, entries = parse_flashcards(flashcards_path)
        flashcards_dir = os.path.join(OUTPUT_DIR, flashcards_label(flashcards_path))
        os.makedirs(os.path.join(flashcards_dir, "audio"), exist_ok=True)

        completed_ids = load_completed_ids(flashcards_dir)
        index = find_start_index(entries, completed_ids)

        new_state = {
            "flashcards_path": flashcards_path,
            "flashcards_dir": flashcards_dir,
            "lang1_col": lang1_col,
            "lang2_col": lang2_col,
            "entries": entries,
            "index": index,
        }

        return render_entry(new_state)

    def render_entry(current_state):
        entries = current_state["entries"]
        index = current_state["index"]
        total = len(entries)

        if index >= total:
            progress = f"**All {total} entries completed.**"
            return (
                current_state, progress,
                "", "", "", "", None, None, None, None,
            )

        entry = entries[index]
        progress = f"Entry **{index + 1} / {total}** (id: `{entry['id']}`)"

        l1 = lang_prompt_label(current_state["lang1_col"])
        l2 = lang_prompt_label(current_state["lang2_col"])

        return (
            current_state,
            progress,
            f"### {l1}", entry["text1"],
            f"### {l2}", entry["text2"],
            None, None, None, None,
        )

    def do_replay(recorded_audio):
        return recorded_audio

    def do_save(current_state, edited_text1, edited_text2, rec_audio1, rec_audio2):
        entries = current_state["entries"]
        index = current_state["index"]

        if index >= len(entries):
            return (current_state, *render_entry(current_state)[1:], "Nothing left to save.")

        if not rec_audio1 or not rec_audio2:
            progress_vals = render_entry(current_state)
            return (current_state, *progress_vals[1:], "Record both languages before saving.")

        entry = entries[index]
        entry["text1"] = edited_text1
        entry["text2"] = edited_text2

        flashcards_dir = current_state["flashcards_dir"]
        lang1_col = current_state["lang1_col"]
        lang2_col = current_state["lang2_col"]

        dest1 = audio_path(flashcards_dir, entry["id"], lang1_col)
        dest2 = audio_path(flashcards_dir, entry["id"], lang2_col)
        shutil.copyfile(rec_audio1, dest1)
        shutil.copyfile(rec_audio2, dest2)

        append_flashcard(
            flashcards_dir, lang1_col, lang2_col, entry,
            os.path.relpath(dest1, flashcards_dir),
            os.path.relpath(dest2, flashcards_dir),
        )

        current_state["index"] = index + 1
        result = render_entry(current_state)
        return (current_state, *result[1:], f"Saved entry `{entry['id']}`.")

    def do_skip(current_state):
        entries = current_state["entries"]
        if current_state["index"] < len(entries):
            current_state["index"] += 1
        result = render_entry(current_state)
        return (current_state, *result[1:], "Skipped entry.")

    load_btn.click(
        do_load,
        inputs=[flashcards_dropdown, state],
        outputs=[state, progress_md, lang1_label, text1, lang2_label, text2, audio1, audio2, replay1, replay2],
    )

    audio1.change(do_replay, inputs=[audio1], outputs=[replay1])
    audio2.change(do_replay, inputs=[audio2], outputs=[replay2])

    save_btn.click(
        do_save,
        inputs=[state, text1, text2, audio1, audio2],
        outputs=[state, progress_md, lang1_label, text1, lang2_label, text2, audio1, audio2, replay1, replay2, status_md],
    )

    skip_btn.click(
        do_skip,
        inputs=[state],
        outputs=[state, progress_md, lang1_label, text1, lang2_label, text2, audio1, audio2, replay1, replay2, status_md],
    )

    if flashcards_paths:
        demo.load(
            do_load,
            inputs=[flashcards_dropdown, state],
            outputs=[state, progress_md, lang1_label, text1, lang2_label, text2, audio1, audio2, replay1, replay2],
        )


if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS)

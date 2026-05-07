from __future__ import annotations

import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


DOCX_PATH = Path("Final_Report.docx")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def get_text(node: ET.Element) -> str:
    return "".join(t.text or "" for t in node.findall(".//w:t", NS)).strip()


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    ppr = paragraph.find("w:pPr", NS)
    for child in list(paragraph):
        if child is not ppr:
            paragraph.remove(child)

    run = ET.Element(w_tag("r"))
    text_el = ET.SubElement(run, w_tag("t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        text_el.set(f"{{{XML_NS}}}space", "preserve")
    text_el.text = text
    paragraph.append(run)


def set_cell_text(cell: ET.Element, text: str) -> None:
    tcpr = cell.find("w:tcPr", NS)
    for child in list(cell):
        if child is not tcpr:
            cell.remove(child)

    para = ET.Element(w_tag("p"))
    run = ET.SubElement(para, w_tag("r"))
    text_el = ET.SubElement(run, w_tag("t"))
    if text.startswith(" ") or text.endswith(" ") or "  " in text:
        text_el.set(f"{{{XML_NS}}}space", "preserve")
    text_el.text = text
    cell.append(para)


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)

    ET.register_namespace("w", W_NS)

    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        document_xml = zin.read("word/document.xml")
        other_files = [(item, zin.read(item.filename)) for item in zin.infolist() if item.filename != "word/document.xml"]

    root = ET.fromstring(document_xml)
    body = root.find("w:body", NS)
    if body is None:
        raise RuntimeError("Could not find document body")

    children = list(body)
    paragraph_positions = {
        "dataset_description": 11,
        "baseline_details": 33,
        "team_collaboration": 115,
        "conclusion_limitations": 121,
    }

    dataset_para = children[paragraph_positions["dataset_description"]]
    baseline_para = children[paragraph_positions["baseline_details"]]
    team_para = children[paragraph_positions["team_collaboration"]]
    conclusion_para = children[paragraph_positions["conclusion_limitations"]]

    if dataset_para.tag != w_tag("p") or "public ISIC 2019 Kaggle dataset" not in get_text(dataset_para):
        raise RuntimeError("Unexpected dataset description paragraph position")
    if baseline_para.tag != w_tag("p") or "This satisfies the project requirement" not in get_text(baseline_para):
        raise RuntimeError("Unexpected baseline details paragraph position")
    if team_para.tag != w_tag("p") or "The team used a shared problem definition" not in get_text(team_para):
        raise RuntimeError("Unexpected team collaboration paragraph position")
    if conclusion_para.tag != w_tag("p") or "The main limitation is that the processed ISIC 2019 setup" not in get_text(conclusion_para):
        raise RuntimeError("Unexpected conclusion paragraph position")

    set_paragraph_text(
        dataset_para,
        "The dataset used in this project is the public ISIC 2019 Kaggle dataset of dermoscopic skin lesion images with diagnostic ground-truth labels and metadata. After excluding UNK, the final dataset contains 25,331 JPG dermoscopic images with CSV label and metadata files. The data were prepared by removing UNK samples, encoding the eight retained classes, and creating stratified train/validation/test splits.",
    )

    set_paragraph_text(
        baseline_para,
        'This satisfies the project requirement that proposed models be trained from scratch. The baseline is trained on the ISIC 2019 training split with an 8-class classification head. Its final test performance is macro precision 0.3849, macro recall 0.4867, and macro F1 0.4048. It provides a transformer-based reference point for comparison against the proposed Swin Transformer and improvement techniques.',
    )

    set_paragraph_text(
        team_para,
        "The team used a shared problem definition, common dataset preparation, and consistent train/validation/test splits so that all transformer approaches were evaluated under the same protocol. All members used the same shared ISIC 2019 split, baseline model, and evaluation setup for fair comparison across architectures. Coordination was handled through weekly Microsoft Teams meetings, shared GitHub repositories for version control, and shared report artifacts so that experiments, metrics, plots, and final written results could be integrated consistently. Collectively, the team produced the shared baseline/foundation comparison setup, the final report, the final presentation, and the presentation video.",
    )

    set_paragraph_text(
        conclusion_para,
        "A key lesson learned is that bias-mitigation methods must be evaluated with both class-imbalance-aware metrics and dataset assumptions in mind: a technique can improve recall while still hurting overall F1, and a bias claim is weaker when explicit ruler annotations are unavailable. The main limitation is that the processed ISIC 2019 setup does not include explicit ruler/no-ruler annotations. Therefore, this project is best interpreted as a study of ruler-like border artifact mitigation rather than a definitive ruler-bias measurement study. Future work should use a curated ruler-annotated test set, tune the strength of border masking and attention regularization, and explore more clinically grounded prompts or fine-tuned medical foundation models. Despite this limitation, the project demonstrates a complete transformer classification pipeline with scratch-trained models, consistent splits, multiple improvement techniques, foundation-model comparison, quantitative metrics, and visual analysis.",
    )

    tables = body.findall("w:tbl", NS)
    if len(tables) < 10:
        raise RuntimeError(f"Expected at least 10 tables, found {len(tables)}")

    contributions_table = tables[9]
    rows = contributions_table.findall("w:tr", NS)
    if len(rows) != 4:
        raise RuntimeError(f"Unexpected contributions table shape: {len(rows)} rows")

    row_updates = {
        1: (
            "Jinali",
            "Implemented and evaluated the Segmented ViT approach, analyzed its three improvement-technique variants, and integrated the Segmented ViT results into the cross-model comparison.",
        ),
        2: (
            "Thanishk",
            "Implemented and evaluated the Swin Transformer approach, including the three ruler/border-mitigation techniques used in this report, and prepared the strongest trained-model analysis.",
        ),
        3: (
            "Tommy",
            "Implemented and evaluated the DeiT approach, analyzed its performance relative to the shared baseline, and contributed DeiT results to the final comparison.",
        ),
    }

    for row_index, (member, contribution) in row_updates.items():
        cells = rows[row_index].findall("w:tc", NS)
        if len(cells) != 2:
            raise RuntimeError(f"Unexpected cell count in row {row_index}: {len(cells)}")
        set_cell_text(cells[0], member)
        set_cell_text(cells[1], contribution)

    tmp_path = DOCX_PATH.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for item, data in other_files:
            zout.writestr(item, data)
        zout.writestr("word/document.xml", ET.tostring(root, encoding="utf-8", xml_declaration=True))

    DOCX_PATH.unlink()
    tmp_path.rename(DOCX_PATH)


if __name__ == "__main__":
    main()

#
# Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering
# Paper: https://arxiv.org/pdf/2306.00526.pdf
# Code: https://github.com/WenjinW/LATIN-Prompt
#

#
# LATIN Prompting
#
def to_prompt(scan, img_size: tuple[int, int]) -> str:
    # Convert ocr bboxes to latin boxes
    w, h = img_size
    texts = [x["text"] or "" for x in scan]
    boxes = [
        [
            int(b["bbox"].TLx * w),
            int(b["bbox"].TLy * h),
            int(b["bbox"].BRx * w),
            int(b["bbox"].BRy * h),
        ]
        for b in scan
    ]

    # Now continue with the latin prompting from https://github.dev/WenjinW/LATIN-Prompt
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0

    while len(boxes) > 0:
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and _is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = _union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]

    max_line_char_num = max(max_line_char_num, 1)
    char_width = line_width / max_line_char_num
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " * (left_char_num - len(space_line_text))
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return "\n".join(space_line_texts)


def _is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """

    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False


def _union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]

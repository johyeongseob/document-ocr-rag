"""Create a synthetic Korean receipt-like image for the OCR lab."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path("sample_document.png")
FONT = Path(r"C:\Windows\Fonts\malgun.ttf")


def main() -> None:
    image = Image.new("RGB", (1000, 720), "white")
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype(str(FONT), 52)
    body = ImageFont.truetype(str(FONT), 34)

    draw.text((70, 55), "넥스인테크놀로지 OCR 실습", font=title, fill="black")
    draw.line((70, 130, 930, 130), fill="black", width=3)

    lines = [
        "문서번호: DOC-2026-0806",
        "고객명: 조형섭",
        "납세자번호(TIN): 123-45-67890",
        "발급일자: 2026년 8월 6일",
        "결제금액: 128,500원",
        "Computer Vision & Document AI",
    ]
    for index, line in enumerate(lines):
        draw.text((80, 175 + index * 72), line, font=body, fill=(25, 25, 25))

    draw.rectangle((60, 155, 940, 635), outline=(90, 90, 90), width=2)
    image.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()

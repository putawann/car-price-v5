# ผลฝึกโมเดลราคารถ v5

ผลรอบล่าสุด 3 งาน (DT / RF / XGB บน merge50minC, ผู้ชนะ = CV MAE ต่ำสุด) จาก output ที่บันทึกไว้

| โมเดล / แบบ | ชุดข้อมูล | Test MAE (บาท) | Test R² | CV MAE ± SD (บาท) |
|---|---|---:|---:|---:|
| DT / base | merge50minC | 56,628 | 0.9319 | 57,931 ± 477 |
| RF / base | merge50minC | 52,275 | 0.9431 | 53,323 ± 256 |
| XGB / base | merge50minC | 51,271 | 0.9458 | 52,256 ± 533 |

ผู้ชนะของ merge50minC: **XGB / base** (test MAE 51,271 บาท)

**รายงานฉบับเต็ม:** [docs/index.html](docs/index.html) · [GitHub Pages](https://putawann.github.io/car-price-v5/)

ผลรอบเก่า 9 งาน (ข้อมูลรุ่นเก่า, กฎ 1-SE, เทียบกับรอบใหม่ไม่ได้): [docs/old.html](docs/old.html)

[ผลรันรอบใหม่และ MAE / RMSE / MAPE ของ 5-fold CV](docs/index.html#latest-results)

ค่าพารามิเตอร์ทั้งหมดและที่มารายไฟล์: [พารามิเตอร์ของแต่ละไฟล์](docs/index.html#parameters)

สร้างซ้ำจากโฟลเดอร์นี้: `python -X utf8 make_report.py` (อ่าน saved outputs/source และ defaults จาก sklearn/xgboost ที่ติดตั้ง ไม่รัน notebook)

เผยแพร่ GitHub Pages จาก branch `main` และโฟลเดอร์ `/docs`

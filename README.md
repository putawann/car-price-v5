# ผลฝึกโมเดลราคารถ v5

สรุปจาก output ที่บันทึกไว้ 12 งาน (DT / RF / XGB × 2 ชุดข้อมูล × base / adj) เลือกผู้ชนะด้วย test MAE ภายในรอบล่าสุดของแต่ละชุดข้อมูล

| ชุดข้อมูล | ผู้ชนะ | รอบ | Test MAE (บาท) | Test R² |
|---|---|---|---:|---:|
| new_cardb | XGB / adj | รอบเก่า (กฎ 1-SE) | 62,883 | 0.9386 |
| merge50minC | XGB / base | รอบใหม่ 2026-10-27 | 51,271 | 0.9458 |

ผลสองรอบปนกัน: 3 งานเป็นรอบใหม่ (ผู้ชนะ = CV MAE ต่ำสุด) และ 9 งานเป็นรอบเก่า (เลือกด้วยกฎ 1-SE) ซึ่งใช้ข้อมูลต่างกัน ผลรอบเก่า: ข้อมูล 29,571 แถว เลือกด้วยกฎ 1-SE ไม่นำมาจัดอันดับร่วมกับรอบใหม่ · ผลรอบเก่า: ข้อมูล 47,001 แถว เลือกด้วยกฎ 1-SE ไม่นำมาจัดอันดับร่วมกับรอบใหม่ · ไม่เทียบ base/adj ข้ามรอบ

**รายงานฉบับเต็ม:** [docs/index.html](docs/index.html) · [GitHub Pages](https://putawann.github.io/car-price-v5/)

[ผลรันรอบใหม่และ MAE / RMSE / MAPE ของ 5-fold CV](docs/index.html#latest-results)

ค่าพารามิเตอร์ทั้งหมดและที่มารายไฟล์: [พารามิเตอร์ของแต่ละไฟล์](docs/index.html#parameters)

สองชุดใช้แถว test ต่างกัน ควรเทียบภายในชุดก่อน รายงานมีผลการปรับข้อมูล, ablation, importance, overfit, learning curves และเวลารัน

สร้างซ้ำจากโฟลเดอร์นี้: `python -X utf8 make_report.py` (อ่าน saved outputs/source และ defaults จาก sklearn/xgboost ที่ติดตั้ง ไม่รัน notebook)

เผยแพร่ GitHub Pages จาก branch `main` และโฟลเดอร์ `/docs`

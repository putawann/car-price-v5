# ผลฝึกโมเดลราคารถ v5

สรุปจาก output ที่บันทึกไว้ 12 งาน (DT / RF / XGB × 2 ชุดข้อมูล × base / adj) เลือกผู้ชนะด้วย test MAE ภายในแต่ละชุดข้อมูล

| ชุดข้อมูล | ผู้ชนะ | Test MAE (บาท) | Test R² |
|---|---|---:|---:|
| new_cardb | XGB / adj | 62,883 | 0.9386 |
| merge50minC | XGB / base | 52,195 | 0.9431 |

**รายงานฉบับเต็ม:** [docs/index.html](docs/index.html) · [GitHub Pages](https://putawann.github.io/car-price-v5/)

ค่าพารามิเตอร์ทั้งหมดและที่มารายไฟล์: [พารามิเตอร์ของแต่ละไฟล์](docs/index.html#parameters)

สองชุดใช้แถว test ต่างกัน ควรเทียบภายในชุดก่อน รายงานมีผลการปรับข้อมูล, ablation, importance, overfit, learning curves และเวลารัน

สร้างซ้ำจากโฟลเดอร์นี้: `python -X utf8 make_report.py` (อ่าน saved outputs/source และ defaults จาก sklearn/xgboost ที่ติดตั้ง ไม่รัน notebook)

เผยแพร่ GitHub Pages: เลือก branch ที่มีรายงานและโฟลเดอร์ `/docs` แล้วแทน `https://putawann.github.io/car-price-v5/` ด้วย URL จริง เผยแพร่เฉพาะไฟล์สคริปต์/รายงานที่อนุญาต และอย่านำ notebook หรือไฟล์ข้อมูลเข้ารายงานสาธารณะ

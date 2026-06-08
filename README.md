1. Install conch
```bash
git clone https://github.com/mahmoodlab/CONCH
cd CONCH
pip install -e .
cd ..
```

2. 
```bash
pip install -r requirements.txt


pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
--index-url https://download.pytorch.org/whl/cu118

```

3. Thay path:
Vào diff/crag_10_uni_conch/configs sửa theo data của mình

Training:

```bash
# xong sẽ tự chạy đánh giá full-image trên tập test
python scripts/train.py --config configs/crag.yaml

# Bỏ qua test sau train:
python scripts/train.py --config configs/crag.yaml --skip-test
```

infererence/test:

```bash
#Chạy test
python scripts/test.py --config configs/crag.yaml

#Chạy vẽ mask cho ảnh /path/to/img.png
python scripts/inference.py --config configs/crag.yaml --weights outputs/checkpoints/crag/seg_crag_uni_conch_best.pt --image /path/to/img.png --out pred.png
```

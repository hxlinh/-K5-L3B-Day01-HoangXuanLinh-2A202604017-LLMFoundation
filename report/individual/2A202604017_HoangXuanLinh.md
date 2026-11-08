# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604017
- **Họ và tên:** Hoàng Xuân Lĩnh
- **SEED** (in ở ô đầu notebook): 3572

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.845      0.911        0.974   
2. BoW + MLP (deep learning)         0.967   0.897      0.930        1.000   
3a. Encoder (MiniLM) + kNN           0.862   0.969      0.913        0.976   
3b. Encoder (MiniLM) + LogReg        0.949   0.969      0.959        0.976   
3c. Encoder (MiniLM) + MLP head      0.989   0.969      0.979        0.976   
4a. LLM nhỏ zero-shot                0.905   0.950      0.927        0.927   
4b. LLM lớn zero-shot                1.000   1.000      1.000        1.000   
4c. LLM nhỏ few-shot                 1.000   1.000      1.000        1.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.964      14293.130       0.040   
2. BoW + MLP (deep learning)        14.936      37512.651       0.158   
3a. Encoder (MiniLM) + kNN          66.689         35.196      25.296   
3b. Encoder (MiniLM) + LogReg       66.862         45.976      19.850   
3c. Encoder (MiniLM) + MLP head     67.336         46.933      19.839   
4a. LLM nhỏ zero-shot                0.000          1.125     786.403   
4b. LLM lớn zero-shot                0.000          1.050     912.016   
4c. LLM nhỏ few-shot                 0.000          1.261     760.770   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.115         0.000  
2. BoW + MLP (deep learning)          0.192         0.000  
3a. Encoder (MiniLM) + kNN           34.951         0.000  
3b. Encoder (MiniLM) + LogReg        26.559         0.000  
3c. Encoder (MiniLM) + MLP head      28.846         0.000  
4a. LLM nhỏ zero-shot              1807.615         0.006  
4b. LLM lớn zero-shot              1143.494         0.057  
4c. LLM nhỏ few-shot               1129.564         0.015  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

1. MLP sẽ tốt hơn Logistic Regression: Đúng là tốt hơn nhưng không nhiều như tôi tưởng
2. Giá khi dùng model lớn đắt hơn model nhỏ 8 lần: Tôi nghĩ model nhỏ sẽ không tuân thủ prompt bằng model lớn nên sẽ tốn token cho việc suy nghĩ nhiều hơn dẫn đến giá như thế nhưng không ngờ vẫn là gấp 10 lần như giá per token
3. Fewshot làm chi phí tăng gần gấp đôi nhưng thực tế là tăng hơn gấp 2.5 lần do tôi không để ý và tưởng chỉ thêm 1 example nên ước chừng sai độ dài prompt.

## 4. Phân tích 50 case khó

BoW suy giảm nhiều nhất: LogReg chỉ đạt 50%, MLP đạt 52%. Hai nhóm khó nhất là spam giống tự nhiên (LogReg 0%, MLP 10%) và né từ khóa (LogReg 17%, MLP 0%).
2 case cụ thể:
- Spam né từ khóa: Thay ký tự bằng số hoặc ký tự có hình dạng tương tự chữ cái đó khiến BoW không nhận diện được từ khóa quen thuộc.
- Tin nhắn nói về spam: MiniLM + kNN chỉ đạt 25% ở nhóm này. Mô hình dễ nhầm tin nhắn đang cảnh báo về spam với tin nhắn thực sự là spam do nội dung có ngữ nghĩa tương tự.

## 5. Thử thách cá nhân

Đề #3: DL · Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?
Cách làm: Sử dụng lại kiến trúc MLP và hàm huấn luyện có sẵn. Lần lượt thay hidden bằng 4 và 512, huấn luyện 30 epochs và đánh giá bằng F1 trên tập test.
Kết quả:
{'Hidden': 4, 'F1 Test': np.float64(0.941), 'Train Time (s)': 1.413}
{'Hidden': 512, 'F1 Test': np.float64(0.926), 'Train Time (s)': 14.44}
Giải thích: Mạng 4 neuron đạt F1 cao hơn và train nhanh hơn khoảng 10 lần. Trong thí nghiệm này, tăng số neuron không giúp cải thiện hiệu quả mà còn làm tăng chi phí tính toán. Vì vậy, nên ưu tiên mạng nhỏ khi dữ liệu và bài toán không quá phức tạp, hoặc khi cần tiết kiệm tài nguyên và thời gian.

## 6. Phương án cho sản phẩm thật

Tôi chọn BoW + MLP, từ bảng so sánh ở mục 2 cho thấy phương án này vừa đáp ứng thời gian yêu cầu vừa tiết kiệm chi phí. Chấp nhận đánh đổi hiệu quả thấp hơn các phương pháp khác như Embed, LLM. 

## 7. AI log và điều đã học

Tôi chủ yếu hỏi AI về các công thức toán, giải thích một số code bạn viết sẵn, hỏi syntax. Tôi đã sửa những phần sai syntax, hiểu được thêm về cách AI học và một số phép biến đổi.

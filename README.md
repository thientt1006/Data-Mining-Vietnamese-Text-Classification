# Data Mining: Vietnamese Text Classification
## Môn học
- Khai thác dữ liệu truyền thông xã hội - IE403.L21
- GV: Nguyễn Văn Kiệt
## Nhóm sinh viên thực hiện
| Họ và tên          | MSSV |
| -------------------|:--------:|
| Trương Thanh Thiên | 18521431 |
| Võ Trung Hiếu      | 18520758 |
| Trần Quốc Thành    | 18521414 |
| Trần Anh Thư       | 18521464 |
| Mai Xuân Tú        | 18521581 |

## Mục lục
1. [Mô tả đồ án](#1-mô-tả-đồ-án)
2. [Dataset](#2-dataset)
    - [UIT-VSMEC](#uit-vsmec)
    - [UIT-VSFC](#uit-vsfc)
    - [UIT-ViCTSD](#uit-victsd)
3. [Các phương pháp](#3-các-phương-pháp)
    - Logistic Regression
    - CNN-LSTM
    - PhoBert
    - Bert4News
    - XLM-R
4. [Kết quả và đánh giá trên các dataset](#4-kết-quả-và-đánh-giá-trên-các-dataset)
    - UIT-VSMEC
    - UIT-VSFC
    - UIT-ViCTSD
  
## 1. Mô tả đồ án
- Nhóm thực hiện phân tích trên ba bộ dữ liệu tiếng Việt, gồm: UIT-VSMEC, UIT-VSFC và UIT-ViCTSD.
- Áp dụng 5 phương pháp: Logistic Regression, CNN-LSTM, PhoBert, Bert4News và XLM-R vào 3 bộ dữ liệu để train model.
- Đánh giá model thông qua các độ đo như: Precision, Recall, Accuracy và F1-Score.
- So sánh kết quả của các độ đo để xem xét đâu là phương pháp phù hợp nhất đối với mỗi bộ dữ liệu.
- Kết quả này được ứng dụng vào việc Xây dựng một ứng dụng phân loại văn bản trên tiếng Việt.

## 2. Dataset
### **UIT-VSMEC**:
- **Đường dẫn bài báo**: https://arxiv.org/abs/1911.09339
- **Mô tả**: Bộ ngữ liệu là dữ liệu bình luận của người dùng trên mạng xã hội đã được thu thập và gán nhãn. Trong bộ ngữ liệu này, biến phân loại là Emotion với miền giá trị là 1 trong số 7 giá trị sau: ‘Other’, ‘Disgust’, ‘Enjoyment’, ‘Anger’, ‘Surprise’, ‘Sadness’, ‘Fear’.
- **Kích thước**:
  - Train: 5548 dòng dữ liệu.
  - Dev: 686 dòng dữ liệu.
  - Test: 693 dòng dữ liệu.
- **Thách thức**: Bộ dữ liệu mang tính thách thức từ nội dung đến hình thức. Bản chất là câu bình luận trên Mạng xã hội nên ngữ liệu có tính phi cấu trúc cao, câu từ và ngữ pháp tự do. Nhiều câu bình luận sai chính tả và có kèm emotion icon, chính 2 tính chất này đã góp phần tăng độ thách thức cho bộ ngữ liệu.
  
  ![Dữ liệu mẫu cho bộ ngữ liệu UIT-VSMEC](https://user-images.githubusercontent.com/66638129/186841304-05ffe880-7768-4a29-9525-0c8ab1c59e03.png)
- Sau khi phân tích dataset và rút ra được một số nhận xét:
  - Phân bố giá trị nhãn của biến Emotion không đồng đều và có số lượng chênh lệch cao giữa các nhãn.
  - Ở tập Train, nhãn có số lượng thấp nhất là 242 (‘Surprise’) trong khi nhãn nhiều nhất là 1558 (‘Enjoyment’).
    
    ![Phân bố tập giá trị nhãn biến Emotion của bộ ngữ liệu UIT-VSMEC](https://user-images.githubusercontent.com/66638129/186841998-5538105c-7d8d-447a-ac8a-16ee26d2a87d.png)

### **UIT-VSFC**: 
- **Đường dẫn bài báo**: https://www.academia.edu/40956009/UIT-VSFC_Vietnamese_Students_Feedback_Corpus_for_Sentiment_Analysis
- **Mô tả**: Bộ ngữ liệu là dữ liệu đánh giá sau môn học của sinh viên với 2 biến phân loại là Sentiments và Topic với miền giá trị lần lượt là {0, 1, 2} cho Sentiments và {0, 1, 2, 3} cho Topic.
- **Kích thước**:
  - Train: 11426 dòng dữ liệu.
  - Dev: 1583 dòng dữ liệu.
  - Test: 3166 dòng dữ liệu.
- **Thách thức**: So với UIT-VSMEC, bộ ngữ liệu này có phần ít thách thức hơn, do bản chất là lời nhận xét và đánh giá của sinh viên sau môn học nên câu từ có tính câu trúc hơn, hiếm có những câu sai chính tả hoặc câu có chứa emotion icon.
    
  ![Dữ liệu mẫu cho bộ ngữ liệu UIT-VSFC](https://user-images.githubusercontent.com/66638129/186842813-75709ebe-071f-4c80-8fc5-b102f510c4a2.png)
- Sau khi phân tích dataset và rút ra được một số nhận xét:
  - Đối với biến *Sentiments*:
    - Phân bố giá trị nhãn của biến Sentiments không đồng đều và có số lượng chênh lệch cao giữa các nhãn.
    - Ở tập Train, nhãn có số lượng thấp nhất là 458 (‘1’) trong khi nhãn nhiều nhất là 5643 (‘2’).
      
      ![Phân bố tập giá trị nhãn biến Sentiments của bộ ngữ liệu UIT-VSFC](https://user-images.githubusercontent.com/66638129/186843048-6c33c69e-fa00-4526-8b36-9a92fae84d1b.png)
  - Đối với biến *Topic*:
    - Phân bố giá trị nhãn của biến Topic không đồng đều và có số lượng chênh lệch cao giữa các nhãn.
    - Ở tập Train, nhãn có số lượng thấp nhất là 497 (‘2’) trong khi nhãn nhiều nhất là 8166 (‘0’).
      
      ![Phân bố tập giá trị nhãn biến Topic của bộ ngữ liệu UIT-VSFC](https://user-images.githubusercontent.com/66638129/186843210-431a91c8-fa10-4944-9e8c-ad7b4dd99523.png)
   
### **UIT-ViCTSD**:
- **Đường dẫn bài báo**: https://arxiv.org/abs/2103.10069
- **Mô tả**: Bộ ngữ liệu là dữ liệu bình luận của người dùng cho 1 bài viết theo từng chủ đề cụ thể. Bộ ngữ liệu này bao gồm 2 biến phân loại là Constructiveness và Toxicity với cùng miền giá trị là {0, 1}.
- **Kích thước**:
  - Train: 7000 dòng dữ liệu.
  - Dev: 2000 dòng dữ liệu.
  - Test: 1000 dòng dữ liệu.
- **Thách thức**: UIT-ViCTSD tương đồng với UIT-VSMEC, bộ ngữ liệu này cũng là dữ liệu bình luận của người dùng trên mạng xã hội, tuy nhiên các bình luận này được thu thập theo từng chủ đề cụ thể.
  
  ![Dữ liệu mẫu cho bộ ngữ liệu UIT-ViCTSD](https://user-images.githubusercontent.com/66638129/186844801-57144641-6070-4d3e-b1be-772f2a221b91.png)
- Sau khi phân tích dataset và rút ra được một số nhận xét:
  - Đối với biến *Sentiments*:
    - Phân bố giá trị nhãn của biến Constructiveness không đồng đều và có số lượng chênh lệch khá cao giữa các nhãn.
    - Ở tập Train, nhãn có số lượng thấp nhất là 2503 (‘1’) trong khi nhãn nhiều nhất là 4497 (‘0’). Tuy nhiên, xét về phân bố thì biến Constructiveness vẫn cân bằng hơn so với biến Toxicity.
      
      ![Phân bố tập giá trị nhãn biến Constructiveness của bộ ngữ liệu UIT-ViCTSD](https://user-images.githubusercontent.com/66638129/186845018-2252d1b3-03b0-4396-b9ac-552c0d9f5043.png)
  - Đối với biến *Toxicity*:
    - Phân bố giá trị nhãn của biến Toxicity không đồng đều và có số lượng chênh lệch rất cao giữa các nhãn.
    - Ở tập Train, nhãn có số lượng thấp nhất là 759 (‘1’) trong khi nhãn nhiều nhất là 6241 (‘0’).
     
      ![Phân bố tập giá trị nhãn biến Toxicity của bộ ngữ liệu UIT- ViCTSD](https://user-images.githubusercontent.com/66638129/186845168-9bee0802-2144-4af3-8619-757773013ded.png)

## 3. Các phương pháp
  - Logistic Regression
  - [CNN-LSTM](https://aclanthology.org/2020.lrec-1.300) 
  - [PhoBert](https://arxiv.org/abs/2003.00744)
  - [Bert4News](https://arxiv.org/abs/2101.12672)
  - [XLM-R](https://arxiv.org/abs/1911.02116)
  
## 4. Kết quả và đánh giá trên các dataset
### **UIT-VSMEC**
  
  ![Bảng so sánh các độ đo giữa các phương pháp có trong bài báo và các phương pháp chạy thực nghiệm trên bộ dữ liệu UIT-VSMEC](https://user-images.githubusercontent.com/66638129/186856122-0b948cba-c44e-4559-a747-b427b528f49b.png)
  -	Ở đây, 6 phương pháp đầu là có trong bài báo, 5 phương pháp sau là chúng tôi chạy thực nghiệm.
  -	Sau khi chạy thực nghiệm thì phương pháp Logistic Regression có độ chính xác thấp nhất với 48.63%. Vì đây là phương pháp truyền thống nên nó chưa đủ độ phức tạp để xử lý bài toán này.
  -	Trong khi đó, phương pháp XLM-R cho độ chính xác cao nhất với 64.94%. Vì đây là phương pháp mới (SOTA) nên nó đủ độ phức tạp để xử lý bài toán này.
  -	Trong số 6 phương pháp được thực nghiệm trong bài báo, phương pháp CNN+word2Vec có độ chính xác cao nhất với accuracy và F1 cùng bằng 59.74% trong khi phương pháp XLM-R của chúng tôi đạt độ chính xác cao nhất là 64.94% đối với accuracy và 62.04% trên F1. Độ chính xác cải thiện 5.20% đối với accuracy và 2.30% đối với F1.

### **UIT-VSFC**

  ![Bảng so sánh các độ đo giữa các phương pháp có trong bài báo và các phương pháp chạy thực nghiệm trên bộ dữ liệu UIT-VSFC](https://user-images.githubusercontent.com/66638129/186857263-676dc6cb-c80c-4cf0-9f1c-e6dd5e8d6f85.png)
  - Đối với nhãn *Sentiment*:
    -	F1 của các phương pháp SOTA không có sự chênh lệch nhiều. Tuy nhiên, phương pháp Logistic Regression và CNN-LSTM có F1 thấp hơn các phương pháp còn lại.
    -	Trong số 4 phương pháp được thực nghiệm trong bài báo, phương pháp Bi-LSTM+Word2Vec có độ chính xác cao nhất với F1 đạt 92.00% trong khi phương pháp PhoBert của chúng tôi đạt độ chính xác cao nhất với F1 là 82.00%. Độ chính xác giảm 10.00% đối với F1.
  - Đối với nhãn *Topic*:
    -	F1 của 3 phương pháp CNN-LSTM, PhoBert, Bert4News không có sự chênh lệch nhiều. Tuy nhiên, phương pháp XLM-R có F1 cao nhất với 88.00% và phương pháp Logistic Regression có F1 thấp nhất với 68.00%.
    -	Trong số 4 phương pháp được thực nghiệm trong bài báo, phương pháp Bi-LSTM+Word2Vec có độ chính xác cao nhất với F1 đạt 89.60% trong khi phương pháp XLM-R của chúng tôi đạt độ chính xác cao nhất với F1 là 88.00%. Độ chính xác giảm 1.60% đối với F1.

### **UIT-ViCTSD**

  ![Bảng so sánh các độ đo giữa các phương pháp có trong bài báo và các phương pháp chạy thực nghiệm trên bộ dữ liệu UIT-ViCTSD](https://user-images.githubusercontent.com/66638129/186857620-bc3ff624-76bb-42fc-afdb-6bff02ef4f6b.png)
  - Đối với nhãn *Constructiveness*:
    - Độ chính xác của các phương pháp SOTA thấp hơn so với phương pháp Logistic Regression và CNN-LSTM.
    -	Trong số 8 phương pháp được thực nghiệm trong bài báo, phương pháp LSTM+fastText có độ chính xác cao nhất với accuracy là 80.00% trong khi phương pháp CNN-LSTM của chúng tôi đạt độ chính xác cao nhất là 80.60% đối với accuracy. Độ chính xác cải thiện 0.60% đối với accuracy.
    -	Phương pháp Our system có độ chính xác cao nhất với F1 là 78.59% trong khi phương pháp CNN-LSTM của chúng tôi đạt độ chính xác cao nhất là 79.46% đối với F1. Độ chính xác cải thiện 0.87% đối với F1.
  - Đối với nhãn *Toxicity*:
    -	Độ chính xác của các phương pháp hầu như không có sự chênh lệch nhiều.
    -	Trong số 8 phương pháp được thực nghiệm trong bài báo, phương pháp Logistic Regression có độ chính xác cao nhất với accuracy là 90.27% trong khi phương pháp CNN-LSTM của chúng tôi đạt độ chính xác cao nhất là 91.00% đối với accuracy. Độ chính xác cải thiện 0.73% đối với accuracy.
    -	Phương pháp Random Forest có độ chính xác cao nhất với F1 là 90.03% trong khi phương pháp PhoBert của chúng tôi đạt độ chính xác cao nhất là 72.92% đối với F1. Độ chính xác giảm 17.11% đối với F1.

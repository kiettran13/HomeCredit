### Default Risk Predict Model — HomeCredit
Trong dự án này này, tôi sẽ cùng dựng một pipeline dự báo **khả năng vỡ nợ** (default) cho bộ dữ liệu Home Credit - một dịch vụ chuyên cung cấp các hạn mức tín dụng (khoản vay) cho dân số không có tài khoản ngân hàng. Mục tiêu của dự án là xây dựng một mô hình cân đối giữa chỉ số kỹ thuật và mục tiêu kinh doanh thực tế, kết quả của mô hình phải ra một **quyết định duyệt/không duyệt** rõ ràng

Dữ liệu để xác lập mục tiêu kinh doanh được tham khảo từ dữ liệu của các tổ chức tín dụng trên thị trường và HomeCredit với mục tiêu cân đối giữa tỷ lệ chấp nhận khoản vay và trần tỷ lệ nợ xấu chấp nhận: 
- Tỷ lệ chấp nhận khoản vay nằm trong khoảng **60%-70%** 
- Trần tỷ lệ chấp nhận nợ xấu là **4.5%**.

Điểm quan trọng nhất của mô hình nằm ở cách chốt ngưỡng quyết định. Thay vì chạy theo một chỉ số đẹp, tôi sẽ chọn một **ngưỡng quyết định** sao cho đáp ứng mục tiêu kinh doanh. Khi đã chốt, ngưỡng này sẽ được khoá lại và dùng nhất quán để đánh giá và tạo submission. 

Dự án sẽ đi theo trình tự sau

1) **Nhìn nhanh dữ liệu và kiểm tra quan hệ khóa** 
2) **Aggregate các bảng lịch sử** về `SK_ID_CURR` (mỗi khách hàng 1 dòng)  
3) **Ghép feature → split data → preprocessing**  
4) Thiết lập **Baseline Model** trước, sau đó **so sánh vài mô hình** dưới cùng policy  
5) **Chốt threshold** và **tạo submission**

## 1) Import thư viện

Trước khi đụng vào dữ liệu, chúng ta chuẩn bị các thư viện cơ bản cho việc xử lý bảng, dựng pipeline và đánh giá mô hình. Ở đây mình cố tình dùng những công cụ quen thuộc của `pandas` và `scikit-learn`.  

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, average_precision_score, classification_report

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
```
</details>

## 2) Load dữ liệu thô

Bây giờ hãy đọc các bảng dữ liệu Home Credit bao gồm 7 nguồn dữ liệu khác nhau:

* application_train/application_test: dữ liệu huấn luyện và kiểm tra chính với thông tin về mỗi đơn xin vay tại Home Credit. Mỗi khoản vay có hàng riêng và được xác định bởi đặc trưng `SK_ID_CURR`. Dữ liệu đơn xin vay huấn luyện đi kèm với `TARGET` cho biết 0: khoản vay đã được trả hoặc 1: khoản vay không được trả. 
* bureau: dữ liệu về các khoản tín dụng trước đây của khách hàng từ các tổ chức tài chính khác. Mỗi khoản tín dụng trước đây có hàng riêng trong bureau, nhưng một khoản vay trong dữ liệu đơn xin có thể có nhiều khoản tín dụng trước đây.
* bureau_balance: dữ liệu hàng tháng về các khoản tín dụng trước đây trong bureau. Mỗi hàng là một tháng của một khoản tín dụng trước đây, và một khoản tín dụng trước đây có thể có nhiều hàng, một hàng cho mỗi tháng của độ dài tín dụng. 
* previous_application: các đơn xin vay trước đây tại Home Credit của các khách hàng có khoản vay trong dữ liệu đơn xin. Mỗi khoản vay hiện tại trong dữ liệu đơn xin có thể có nhiều khoản vay trước đây. Mỗi đơn xin trước đây có một hàng và được xác định bởi đặc trưng `SK_ID_PREV`. 
* POS_CASH_BALANCE: dữ liệu hàng tháng về các khoản vay điểm bán hàng hoặc tiền mặt trước đây mà khách hàng đã có với Home Credit. Mỗi hàng là một tháng của một khoản vay điểm bán hàng hoặc tiền mặt trước đây, và một khoản vay trước đây có thể có nhiều hàng.
* credit_card_balance: dữ liệu hàng tháng về các thẻ tín dụng trước đây mà khách hàng đã có với Home Credit. Mỗi hàng là một tháng của số dư thẻ tín dụng, và một thẻ tín dụng có thể có nhiều hàng.
* installments_payment: lịch sử thanh toán cho các khoản vay trước đây tại Home Credit. Có một hàng cho mỗi khoản thanh toán đã thực hiện và một hàng cho mỗi khoản thanh toán bị bỏ lỡ.

<details>
<summary><b>code </b></summary>

```
# Load dữ liệu

train = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/application_train.csv")
test  = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/application_test.csv")

bureau = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/bureau.csv")
bureau_balance = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/bureau_balance.csv")
pos = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/POS_CASH_balance.csv")
prev = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/previous_application.csv")
inst = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/installments_payments.csv")
cc = pd.read_csv("/Users/thekiet/Downloads/HomeCredit_Project/credit_card_balance.csv")   

print("train:", train.shape, "test:", test.shape)
print("bureau:", bureau.shape, "bureau_balance:", bureau_balance.shape)
print("pos:", pos.shape, "prev:", prev.shape)
print("inst:", inst.shape)
print("cc:", cc.shape)
```
</details>


<details>summary><b> Output code </b></summary>

```
train: (307511, 122) test: (48744, 121)
bureau: (1716428, 17) bureau_balance: (27299925, 3)
pos: (10001358, 8) prev: (1670214, 37)
inst: (13605401, 8)
cc: (3840312, 23)
```
</details>

### Nhận xét nhanh

- Bảng chính `application_train` có **307,511 dòng và 122 cột**, còn `application_test` có **48,744 dòng và 121 cột**. Đây là phần cốt lõi của hồ sơ vay nên mỗi hồ sơ tương ứng một dòng.

- Các bảng lịch sử thì lớn hơn nhiều: `bureau` có **1,716,428 dòng** và `bureau_balance` lên tới **27,299,925 dòng**. Tương tự, POS, installments, credit card balance cũng rất lớn vì đây là dữ liệu theo thời gian

=> Không thể ghép thẳng các bảng lịch sử vào bảng chính. Ta sẽ phải **gộp lịch sử** thành các đặc trưng gọn trước, rồi mới ghép về `SK_ID_CURR`. Làm đúng bước này sẽ giúp pipeline sạch và tránh lỗi logic, một lựa chọn rất an toàn.

## 3) Xác lập mục tiêu kinh doanh & Kiểm tra tính hợp lý dữ liệu

<details>
<summary><b>Xem toàn bộ code </b></summary>

```
# Business targets + sanity checks

# ===== Business constraints =====     
NPL_CAP_APPROVED = 0.045     # NPL cap cho nhóm approve (tham chiếu)
APPROVAL_TARGET_LOW = 0.6    # Mục tiêu tỷ lệ duyệt vay (60%-70%)
APPROVAL_TARGET_HIGH = 0.7

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"

print("=== BUSINESS TARGETS ===")
print(f"NPL cap (approved):       {NPL_CAP_APPROVED:.2%}")
print(f"Approval target:          {APPROVAL_TARGET_LOW:.0%}–{APPROVAL_TARGET_HIGH:.0%}")

# ===== Main table checks =====
print("\n=== MAIN TABLE CHECKS: train ===")
print("train shape:", train.shape)

# Target distribution
y = train[TARGET_COL]
print("\nTARGET distribution:")
print(y.value_counts(dropna=False))
base_npl = y.mean()
print(f"Base default rate (TARGET=1): {base_npl:.4%}")

# Unique key check
n_unique = train[ID_COL].nunique()
print(f"\nUnique {ID_COL}: {n_unique} / {len(train)}")
print("Duplicate keys in train:", len(train) - n_unique)

# Missing rate overview (top 20)
feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
miss_rate = train[feature_cols].isna().mean().sort_values(ascending=False)

print("\nTop 20 missing-rate features (%):")
print((miss_rate.head(20) * 100).round(2).to_string())

# Quick type overview
cat_cols = train[feature_cols].select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in feature_cols if c not in cat_cols]
print("\nFeature types:")
print(f"Total features: {len(feature_cols)} | Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

print("\n=== MAIN TABLE CHECKS: test ===")
print("test shape:", test.shape)
print(f"Unique {ID_COL} in test: {test[ID_COL].nunique()} / {len(test)}")
print("Duplicate keys in test:", len(test) - test[ID_COL].nunique())

# ===== Relationship sanity checks =====
def rel_check(df, key, name, head=5):
    dup = df.duplicated(key).sum()
    nunq = df[key].nunique()
    print(f"{name}: shape={df.shape} | unique {key}={nunq} | duplicated rows by {key}={dup}")
    if dup == 0:
        print(f"-> Looks like 1-1 on {key}.")
    else:
        # show some most frequent keys
        top = df[key].value_counts().head(head)
        print("-> Top repeated keys:")
        print(top.to_string())

print("\n=== RELATIONSHIP CHECKS ===")
rel_check(bureau, "SK_ID_CURR", "bureau by SK_ID_CURR") # Một khách hàng có thể có nhiều khoản vay tại các TCTD khác 
rel_check(bureau, "SK_ID_BUREAU", "bureau by SK_ID_BUREAU") #Mỗi dòng bureau phải đại diện cho 1 khoản vay CIC 

rel_check(bureau_balance, "SK_ID_BUREAU", "bureau_balance by SK_ID_BUREAU") #Một khoản vay CIC có thể có lịch sử theo nhiều tháng 

rel_check(prev, "SK_ID_CURR", "previous_application by SK_ID_CURR") #Một khách hàng có thể từng apply nhiều khoản vay tại Home Credit 
rel_check(prev, "SK_ID_PREV", "previous_application by SK_ID_PREV") #Mỗi application có phải 1 record duy nhất 

rel_check(pos, "SK_ID_PREV", "POS_CASH_balance by SK_ID_PREV") #Một khoản vay có gồm nhiều tháng thanh toán ở pos hoặc bằng tiền mặt
rel_check(inst, "SK_ID_PREV", "installments_payments by SK_ID_PREV") #Một khoản vay gồm nhiều tháng thanh toán ở HomeCredit
rel_check(cc, "SK_ID_PREV", "credit_card_balance by SK_ID_PREV") # Một khoản vay trên thẻ có nhiều tháng thanh toán
```
</details>

### Nhận xét nhanh

Ở đây chúng ta thấy base default rate khoảng **8.0729%**. Con số này nói lên rằng dữ liệu bị lệch, tức là số ca không vỡ nợ nhiều hơn nhiều so với số ca vỡ nợ.

Nhiều biến có tỷ lệ thiếu cao (trên 60%), chủ yếu liên quan đến nhà ở, phản ánh thực tế nghiệp vụ hơn là lỗi dữ liệu
=> Missing value cần được xử lý như một tín hiệu thông tin, không drop bừa bãi

Các bảng phụ đều thể hiện đúng quan hệ 1–N 
- bureau theo SK_ID_CURR có rất nhiều duplicated → đúng kỳ vọng vì 1 khách có thể có nhiều khoản vay ở tổ chức khác.
- bureau theo SK_ID_BUREAU là 1–1 → hợp lý vì mỗi SK_ID_BUREAU đại diện cho một khoản tín dụng cụ thể.
- bureau_balance theo SK_ID_BUREAU là 1–N rất lớn → đúng nghiệp vụ vì một khoản tín dụng có nhiều dòng theo tháng (lịch sử trạng thái).
- previous_application theo SK_ID_CURR là 1–N → khách có thể apply nhiều lần.
- previous_application theo SK_ID_PREV là 1–1 → mỗi đơn vay là một record.
- POS / installments / credit_card theo SK_ID_PREV đều là 1–N → đúng vì mỗi khoản vay/thẻ có nhiều kỳ trả góp theo tháng.













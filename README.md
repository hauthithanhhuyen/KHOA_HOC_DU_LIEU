# KHOA_HOC_DU_LIEU
# BTVN_TINH DIEM TB
# LINK YOUTUBE:https://youtu.be/eRuTWlQ6caQ
HẦU THỊ THANH HUYỀN
K225480106027
K58KTP
YÊU CẦU:
TỪ FILE ĐIỂM CỦA LỚP,PHÂN CỤM 3 NHÓM
NỘP FILE CODE
VIDEO CHẠY CHƯƠNG TRÌNH
# CHƯƠNG TRÌNH CODE
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_excel("bangdiem.xlsx", header=None, keep_default_na=False)

# tìm đúng dòng MSSV, tên sinh viên, dòng tiêu đề môn
row_mssv = df[df.apply(lambda r: r.astype(str).str.contains("MSSV", case=False).any(), axis=1)].index[0]
row_ten = df[df.apply(lambda r: r.astype(str).str.contains("Tên Sinh Viên", case=False).any(), axis=1)].index[0]
row_mon = df[df.apply(lambda r: r.astype(str).str.contains("Tên Môn học", case=False).any(), axis=1)].index[0]

# cột bắt đầu điểm sinh viên
col_start = df.iloc[row_mssv].astype(str).tolist().index("MSSV") + 1

mssv = df.iloc[row_mssv, col_start:].reset_index(drop=True)
ten = df.iloc[row_ten, col_start:].reset_index(drop=True)

# lấy điểm các môn
scores = df.iloc[row_mon + 1:, col_start:].reset_index(drop=True)

def xu_ly_diem(x):
    x = str(x).strip()

    # môn không học -> bỏ qua
    if x in ["NaN", "nAn", "—"]:
        return np.nan

    # chưa có điểm -> random
    if x == "" or x.lower() == "x":
        return round(np.random.uniform(2.0, 4.0), 1)

    x = x.replace(",", ".")

    try:
        return float(x)
    except:
        return np.nan

scores = scores.map(xu_ly_diem)

# điểm TK(4) trung bình để phân cụm
diem_tk4 = scores.mean(axis=0, skipna=True)

X = diem_tk4.values.reshape(-1, 1)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

centers = kmeans.cluster_centers_.flatten()
sorted_idx = np.argsort(centers)

mapping = {
    sorted_idx[0]: "Khá",
    sorted_idx[1]: "Giỏi",
    sorted_idx[2]: "Xuất sắc"
}

result = pd.DataFrame({
    "MSSV": mssv,
    "Tên sinh viên": ten,
    "Điểm TK(4)": np.round(diem_tk4.values, 2),
    "Cụm": [mapping[c] for c in clusters]
})

# tách từng cụm thành nhóm cột riêng
kha = result[result["Cụm"] == "Khá"][["MSSV", "Tên sinh viên", "Điểm TK(4)"]].reset_index(drop=True)
gioi = result[result["Cụm"] == "Giỏi"][["MSSV", "Tên sinh viên", "Điểm TK(4)"]].reset_index(drop=True)
xuatsac = result[result["Cụm"] == "Xuất sắc"][["MSSV", "Tên sinh viên", "Điểm TK(4)"]].reset_index(drop=True)

ketqua = pd.concat(
    [
        kha.add_prefix("Khá - "),
        gioi.add_prefix("Giỏi - "),
        xuatsac.add_prefix("Xuất sắc - ")
    ],
    axis=1
)

ketqua.to_excel("ketqua_phanCum.xlsx", index=False)

print("Đã xuất file ketqua_phanCum.xlsx đúng định dạng")

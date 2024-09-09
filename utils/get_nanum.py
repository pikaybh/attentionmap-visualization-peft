import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request

# 나눔고딕 폰트 다운로드
font_url = "https://github.com/naver/nanumfont/releases/download/v1.0/NanumFont_TTF_ALL.zip"
font_zip_path = "/tmp/NanumFont.zip"
urllib.request.urlretrieve(font_url, font_zip_path)

# 압축 해제
import zipfile
with zipfile.ZipFile(font_zip_path, 'r') as zip_ref:
    zip_ref.extractall("/tmp/NanumFont")

# 폰트 설정
font_path = "/tmp/NanumFont/NanumGothic.ttf"
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()

# 이후 시각화 코드에서 한글이 올바르게 표시됩니다.

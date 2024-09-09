import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 설치된 폰트 목록에서 한글 폰트를 찾기
font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')

# 한글 폰트를 찾아서 설정 (예: AppleGothic, Malgun Gothic, etc.)
for font_path in font_list:
    if 'NanumGothic' in font_path or 'AppleGothic' in font_path or 'Malgun' in font_path:
        fontprop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = fontprop.get_name()
        break
else:
    print("한글 폰트를 찾을 수 없습니다. 시스템에 한글 폰트가 설치되어 있는지 확인하세요.")

# 위 코드가 성공적으로 폰트를 설정했다면, 이후 시각화 코드에서 한글이 올바르게 표시됩니다.

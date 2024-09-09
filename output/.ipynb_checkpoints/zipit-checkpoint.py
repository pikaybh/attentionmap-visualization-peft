from typing import Optional
import zipfile
import fire
import os


def main(input_dir: str, output: Optional[str] = 'output_images.zip') -> None:
    '''
    Args:
        input_dir(str): 이미지 파일들이 있는 폴더 경로
        output(Optional[str]): 생성될 ZIP 파일 이름
    '''
    # 이미지 파일의 확장자 목록 (필요에 따라 확장자를 추가/제거할 수 있습니다)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Ensure the output directory exists
    output_path = os.path.join(input_dir, output)
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ZipFile 객체 생성 (쓰기 모드로)
    with zipfile.ZipFile(output_path, 'w') as zipf:
        # 폴더 내 모든 파일을 순회
        for root, _, files in os.walk(input_dir):
            for file in files:
                # 파일 확장자 검사
                if os.path.splitext(file)[1].lower() in image_extensions:
                    # 파일 경로
                    file_path = os.path.join(root, file)
                    # ZIP 파일에 추가 (상대 경로로 추가)
                    zipf.write(file_path, os.path.relpath(file_path, input_dir))
    
# Main
if __name__ == "__main__":
    fire.Fire(main)
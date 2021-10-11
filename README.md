# Aegis-ML

### [Microsoft Malware Classification Challenge (BIG 2015)](https://www.kaggle.com/c/malware-classification)

* Envs  

docker를 이용한 container 환경에서 개발되었으므로 docker hub에서 image를 받아 실행하는 걸 권장합니다.

1. 터미널에서 아래 명령어를 실행합니다.

2. `docker pull lwamuhaji/pytorch:2.0` - docker hub에서 이미지를 다운받습니다. (약 13GB)  

3. `docker run -it lwamuhaji/pytorch /bin/bash` - 다운 받은 이미지를 컨테이너로 올리고 /bin/bash를 실행합니다.  

4. 정상적으로 컨테이너가 올라가고 bash가 켜지면 `cd /ML`로 작업폴더로 이동한 후 `python main.py`을 실행해봅니다.

---

- 도커를 사용하지 않는 경우 간단하게 colab을 통해 실행해보실 수 있습니다. [COLAB](https://colab.research.google.com/drive/1sQIq1OeM0tboYkY6LjjYUlo7m3zDN4bh) 이 경우 [이 링크](https://drive.google.com/drive/folders/14GcS14aL7oaUbH5Ta6Qc7hz12U7Y8PEY?usp=sharing)에서 kucis_dataset 폴더의 바로가기를 자신의 구글 드라이브에 추가하고, colab과 google drive를 mount 해야합니다.

- 로컬환경에서 실행하는 경우 anaconda로 pytorch 및 tqdm, matplotlib, pandas, numpy, torchvision, torchsummary 등의 패키지를 설치합니다. 그리고 [이 링크](https://drive.google.com/file/d/1-97yHevn9gdJ_9rLoDd_TT6HLcxkOXtn/view?usp=sharing)에서 kucis_dataset.7z을 다운받은 뒤 압축을 풀고 jupyter notebook 또는 visual studio code로 실행 가능하지만 권장하지 않습니다.


### Converter  

input 폴더에 전처리 전 파일을 넣고 converter.py 를 실행하면 output 폴더에 전처리된 numpy 파일이 생성됩니다.  

Incremental coordinate, size: 256x256 channel: 1

### Gmail Crawler

구글 클라우드 플랫폼에서 gmail api oauth token을 생성하면 그 계정에 한해 메일 정보를 크롤링 할 수 있습니다.

데이터는 크롤러가 실행된 위치의 mails 폴더 내에 저장되고, 메일마다 별도의 폴더가 생성되며 그 안에 메일 내용과 이미지, 첨부파일이 다운로드됩니다.

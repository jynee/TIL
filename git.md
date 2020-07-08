# Git

> Git은 분산형 버전 관리 시스템(DVCS) 중 하나이다.



## 0. Git 기초 설정

* windows 환경에서는 `git for windows`로 검색하여 `git bash`를 설치한다. [다운로드 링크](https://gitforwindows/org/)

* 최초에 컴퓨터에서 git을 사용하는 경우 아래의 설정을 진행한다.

  ```bash
  $ git config --global user.email 이메일 주소
  $ git config --global user.name 유저 네임
  # 확인
  $ git config --global -l
  ```

  * 이메일 주소를 설정할 때, github에 가입된 이메일로 설정을 해야 커밋 이력이 github에 기록된다. 



## 1. Git을 통한 버전관리 기본 흐름

### 1.1. Git 저장소 초기화

> 특정 폴더를 git 저장소로 활용하기 위해서 최초에 입력하는 명령어

```bash
$ git init
Initialized empty Git repository in C:/Users/jynee/Desktop/TIL/.git/
(master) $
```

* .git 폴더가 숨긴 폴더로 생성되며, git bash에서는 `(master)` 라고 표기된다.

* 반드시 git으로 활용되고 있는 폴더 아래에서 저장소를 선언하지 말자!

* 잘못 저장했으면 .git 폴더 삭제!

  

### 1.2. add

> 커밋 대상 파일들을 추가한다.

add 전 상황

``` bash
$ git status
On branch master
# 커밋 없음

No commits yet
# 트랙킹 되지 않는 파일들
# => 새로 생성된 파일이고, git으로 관리 중이지 않는 파일

Untracked files:
	# git add 파일
	# 커밋이 될 것들을 포함시키기 위해서 위의 명령어를 써라! 
  (use "git add <file>..." to include in what will be committed)
        git.md
        markdown-images/
        markdown.md

nothing added to commit but untracked files present (use "git add" to track)

```


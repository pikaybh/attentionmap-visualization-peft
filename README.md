# My Personal Repo for PEFT Llama HF

## Hierachy
                                            
```
.
├── archive
│   ├── model
│   └── output_20240903
├── dataset                                   # 학습 데이터셋 모음집
│   └── json                                  # 제이슨 포맷 데이터셋
│       ├── baemin                            # 데이터셋 1
│       └── csi_report                        # 데이터셋 2
│           └── csi_report_total
├── logs
├── models                                    # 에프티 모델
│   ├── Llama-3-Open-Ko-8B-Baemin-pikaybh     # 에프티 모델 1
│   │   └── results                           # 체크포인트
│   │       ├── checkpoint-1000
│   │       ├── checkpoint-1500
│   │       ├── checkpoint-2000
│   │       ├── checkpoint-2500
│   │       ├── checkpoint-3000
│   │       ├── checkpoint-3500
│   │       ├── checkpoint-4000
│   │       ├── checkpoint-4500
│   │       └── checkpoint-5000
│   │                ...
│   └── Llama-3-Open-Ko-8B-csi-report-acctyp  # 에프티 모델 2
│       ├── ckpts                             # 체크포인트
│       │   ├── checkpoint-1000
│       │   ├── checkpoint-1500
│       │   ├── checkpoint-2000
│       │   ├── checkpoint-2500
│       │   ├── checkpoint-3000
│       │   ├── checkpoint-3500
│       │   ├── checkpoint-4000
│       │   ├── checkpoint-4500
│       │   └── checkpoint-5000
│       │            ...
│       └── dataset                           # 트레인 테스트 셋 나눈거 저장되는 곳
├── output                                    # 어탠션 맵 저장되는 곳
│   └── 2024_09_01_000000
└── utils
```

---

## Author

- **Byunghee Yoo** - [@pikaybh](https://github.com/pikaybh)
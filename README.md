# NEOWISE Young Stellar Object variability analyses 

![스크린샷 2022-06-14 오후 8.34.36](readme_images/paper_resize.png)

🔭 💫 석사 과정 중 진행했던 연구인 우주망원경 WISE를 이용한 원시성 적외선 밝기 데이터 분석 프로젝트입니다. 연구 결과는 논문으로 Astrophysical Journal에 기고하였습니다. [링크](https://ui.adsabs.harvard.edu/abs/2021ApJ...920..132P/abstract) 



📊 여기에서는 ''천문학''적인 설명보다는 "데이터 분석 과정"에 초점을 맞춰 분석 결과를 설명하고자 합니다.

## Introduction

![Jets Protrude from a Young Stellar Object](readme_images/80944_web_resize.jpg)

약 50억년으로 추정되는 태양의 나이에 비해 원시성 (Young Stellar Objects, YSOs)은 태어난지 1000만년이 되지 않는 아주 어린 별이며, 아기가 쑥쑥 자라듯이 격렬한 질량 축적이 이루어지고 이 과정에서 **밝기가 빠르고 불규칙하게 변하는 특징**을 가지고 있습니다.

이러한 원시성의 밝기 변화를 통해 원시 항성계 혹은 태양계의 초기 생성 원리를 밝혀 내려는 연구가 비교적 최근부터 활발히 연구되어지고 있습니다. 본 연구에서는 알려진 5400개의 원시성의 밝기 변화를 적외선 우주망원경인 WISE의 NEOWISE 프로젝트 관측 시계열 데이터로 수집하였습니다.

수집한 시계열 데이터는 outlier 제거 등 분석에 용이한 형태로 정제하였고 이로부터 밝기의 변화량, 표준편차, 주기 등 각각 별에 대한 feature를 계산하였습니다. 

그 후, 계산된 feature들을 이용하여 별의 밝기변화 형태를 Linear, Curved, Periodic, Burst, Drop, Irregular (선형, 곡선형, 주기형, 폭발 - 순간적인 밝기 증가, 깜빡임 - 순간적인 밝기 감소, 불규칙) 의 6가지 형태로 분류하는 데 성공하였습니다. 

<img src="readme_images/vartypes.png" alt="Variable Types"/>

각각의 **원시성의 진화 단계는 선행 연구들로 인해 알려진 상태**이기 때문에, 밝기 변화 분류 결과와 진화 단계를 비교하면 **진화 단계별 밝기 변화의 특징을 파악**할 수 있습니다.  
아래는 가장 초기 단계부터 Class 0/I - Class II - Class III 으로 나누어진 원시성 진화 단계별 변화 형태의 분포표입니다. 괄호 () 는 전체 샘플 대비 % 인데, Class 0 단계에서 밝기 변화율이 54.8%로 가장 큰 것을 알 수 있습니다.

<img src="readme_images/table2.png" alt="table2"/>

본 연구를 통해 원시성 각각의 알려진 **진화 단계**와 새롭게 분류된 **밝기 변화 형태**를 비교 분석한 결과 **더 초기 단계의 원시성일 수록 크고 선형적인 밝기 변화를 보이는 것이 밝혀졌습니다.** 



## Section
















&lt;p align="center"&gt;
  &lt;img src="https://capsule-render.vercel.app/api?type=waving&color=0:667eea,100:764ba2&height=200&section=header&text=2030%20AI%20Otomasyon%20Risk%20Haritasi&fontSize=45&fontAlignY=35&desc=Machine%20Learning%20Bootcamp%20Projesi&descAlignY=55&descAlign=50" /&gt;
&lt;/p&gt;

&lt;p align="center"&gt;
  &lt;img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" /&gt;
  &lt;img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" /&gt;
  &lt;img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" /&gt;
  &lt;img src="https://img.shields.io/badge/R%C2%B2%20Score-84.9%25-00C851?style=for-the-badge" /&gt;
  &lt;img src="https://img.shields.io/badge/CRISP--DM-Metodoloji-blueviolet?style=for-the-badge" /&gt;
&lt;/p&gt;

&lt;p align="center"&gt;
  &lt;img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=22&duration=4000&pause=1000&color=667EEA&center=true&vCenter=true&width=600&lines=Yapay+Zeka+ve+Otomasyon+Analizi;Mesleklerin+Gelecegi+Tahmini;Lineer+Regresyon+Modeli;Veri+Bilimi+Projesi" /&gt;
&lt;/p&gt;

## Proje Ozeti

Bu proje, **2030 yilina kadar** farkli mesleklerin yapay zeka ve otomasyon karsisindaki risk seviyelerini **makine ogrenmesi** ile tahmin etmektedir.

### Metodoloji: CRISP-DM

| Adim | Aciklama | Durum |
|:----:|:---------|:------|
| 1 | Is Anlayisi | Tamamlandi |
| 2 | Veri Anlayisi | Tamamlandi |
| 3 | Veri Hazirlama | Tamamlandi |
| 4 | Modelleme | Lineer Regresyon |
| 5 | Degerlendirme | R2: 84.9% |
| 6 | Raporlama | Tamamlandi |

## Model Performansi

- **R2 Score:** 84.9%
- **MAE:** 0.08
- **Model:** Lineer Regresyon

## Ana Bulgular

### Yuksek Risk (70%+)
| Sektor | Risk | AI Maruziyeti |
|:-------|:----:|:-------------:|
| Insaat | 85% | 30% |
| Lojistik | 82% | 25% |
| Perakende | 78% | 35% |
| Uretim | 75% | 40% |

### Dusuk Risk (25%-)
| Sektor | Risk | AI Maruziyeti |
|:-------|:----:|:-------------:|
| Saglik | 15% | 85% |
| Egitim | 12% | 90% |
| Akademi | 18% | 80% |
| Ar-Ge | 22% | 75% |

## Kurulum

```bash

git clone https://github.com/KULLANICIADIN/ml-bootcamp-ai-risk-map.git

cd ml-bootcamp-ai-risk-map

pip install -r requirements.txt

jupyter notebook notebooks/analysis.ipynb

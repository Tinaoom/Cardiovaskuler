﻿# Cardiovaskuler
# Cardiovaskuler
# Kardiyovasküler Hastalık Risk Tahmini için Yapay Zeka Modeli

Bu depo, kardiyovasküler hastalık riskini tahmin etmek için geliştirilmiş bir yapay zeka modelini içermektedir. Proje, makine öğrenimi tekniklerini kullanarak hasta verilerinden risk faktörlerini analiz etmeyi ve olası kardiyovasküler olayları önceden belirlemeyi amaçlamaktadır.

## İçindekiler

- [Proje Açıklaması](#proje-açıklaması)
- [Teknolojiler](#teknolojiler)
- [Veri Seti](#veri-seti)
- [Model](#model)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

## Proje Açıklaması

Bu proje, kardiyovasküler hastalıkların erken teşhisine ve önlenmesine yardımcı olmak amacıyla geliştirilmiştir. Yapay zeka modelleri, büyük miktardaki tıbbi veriyi analiz ederek, geleneksel yöntemlerle zor tespit edilebilecek örüntüleri ve risk faktörlerini belirleyebilir. Bu proje kapsamında geliştirilen model, çeşitli demografik, yaşam tarzı ve tıbbi geçmiş verilerini kullanarak bireylerin kardiyovasküler hastalık geliştirme olasılığını tahmin etmektedir.

## Teknolojiler

Bu proje aşağıdaki teknolojiler kullanılarak geliştirilmiştir:

- **Python:** Temel programlama dili olarak kullanılmıştır.
- **Pandas:** Veri manipülasyonu ve analizi için kullanılmıştır.
- **NumPy:** Bilimsel hesaplamalar için kullanılmıştır.
- **Scikit-learn:** Makine öğrenimi algoritmaları ve model değerlendirmesi için kullanılmıştır.
- **CatBoost:** Yüksek performanslı gradyan artırma algoritması model geliştirmede kullanılmıştır.
- **Flask:** Web uygulaması oluşturmak ve modeli bir API üzerinden sunmak için kullanılmıştır (eğer web uygulaması varsa).
- **Pickle:** Eğitilmiş modeli kaydetmek ve yüklemek için kullanılmıştır.
- **Jupyter Notebook:** Veri analizi ve model geliştirme süreçlerini göstermek için kullanılmıştır (`cardiovaskuler.ipynb`).

## Veri Seti

Modeli eğitmek için kullanılan veri seti `cardio_train.csv` dosyasında bulunmaktadır. Bu veri seti, kardiyovasküler hastalık geçmişi olan ve olmayan bireylerden toplanmış çeşitli özellikleri içermektedir. Veri setindeki özellikler şunları içerebilir (gerçek veri setinize göre güncelleyin):

- Yaş
- Cinsiyet
- Boy
- Kilo
- Sistolik Kan Basıncı
- Diastolik Kan Basıncı
- Kolesterol seviyesi
- Glikoz seviyesi
- Sigara kullanımı
- Alkol tüketimi
- Fiziksel aktivite durumu
- Vb.

Veri setinin kaynağı ve özellikleri hakkında daha detaylı bilgi için lütfen veri dosyasını inceleyin veya kaynak belirtin.


## Kurulum

Bu projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

1.  **Depoyu klonlayın:**

    ```bash
    git clone [https://github.com/Tinaoom/Cardiovaskuler.git](https://github.com/Tinaoom/Cardiovaskuler.git)
    cd Cardiovaskuler
    ```

2.  **Gerekli kütüphaneleri yükleyin:**

    ```bash
    pip install -r requirements.txt
    ```

    `requirements.txt` dosyası, projenin bağımlı olduğu tüm Python kütüphanelerini listeler.

## Kullanım

Projenin nasıl kullanılacağına dair talimatlar aşağıdadır:

1.  **Eğitilmiş modeli yükleyin:**

    ```python
    import pickle

    with open('cardio_risk_model.pkl', 'rb') as f:
        model = pickle.load(f)
    ```

2.  **Yeni veri ile tahmin yapın:**

    ```python
    import pandas as pd

    # Örnek yeni veri (modelin beklediği özellik sırasına göre düzenleyin)
    new_data = pd.DataFrame({
        'age': [55],
        'gender': [1],
        'cholesterol': [3],
        # Diğer özellikler...
    })

    prediction = model.predict(new_data)
    print(f"Tahmin Edilen Risk: {prediction}")
    ```

3.  **Web uygulamasını çalıştırmak (eğer varsa):**

    ```bash
    python app.py
    ```

    Web uygulaması genellikle `http://localhost:5000` adresinde çalışır.

Daha detaylı kullanım senaryoları ve API endpointleri (eğer varsa) için lütfen ilgili Python dosyalarını (örneğin `app.py`) inceleyin.

## Sonuçlar

Modelin performansı, `cardiovaskuler.ipynb` notebook dosyasında detaylı olarak sunulmuştur. Burada doğruluk, kesinlik, duyarlılık, F1-skoru ve AUC gibi metrikler yer almaktadır. Elde edilen sonuçlar, yapay zeka modellerinin kardiyovasküler hastalık risk tahmininde umut verici bir potansiyele sahip olduğunu göstermektedir.

## Katkıda Bulunma

Bu projeye katkıda bulunmak isterseniz, lütfen aşağıdaki adımları izleyin:

1.  Projeyi fork edin.
2.  Kendi branch'inizi oluşturun (`git checkout -b feature/yeni-ozellik`).
3.  Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`).
4.  Branch'inizi push edin (`git push origin feature/yeni-ozellik`).
5.  Pull request oluşturun.

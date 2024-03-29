# Laporan Proyek Machine Learning - Prediksi Harga Properti

## Domain Proyek

Harga properti merupakan elemen utama dalam pengambilan keputusan dalam proses jual beli properti  [[1](https://www.researchgate.net/publication/353333759_Factors_Influencing_Housing_Purchase_Decision)] [[2](https://www.researchgate.net/publication/285661484_Housing_Market_A_Review_of_Purchase_Decision_of_Potential_Buyers)] [[3](https://www.researchgate.net/publication/354085963_Analysis_of_County_Consumers%27_Housing_Purchase_Intention_and_Influencing_Factors--Based_on_the_Investigation_of_Anyue_County)]. Kemampuan untuk meramalkan harga properti tentunya akan memberikan bantuan kepada calon pembeli sebelum mereka melakukan transaksi jual beli properti.

Dalam konteks transaksi jual beli properti, harga sangat dipengaruhi oleh beberapa faktor, seperti lokasi geografis, demografi, jumlah ruangan, dan faktor-faktor lainnya. Salah satu aspek menarik untuk diselidiki lebih lanjut adalah pandangan atau pemandangan yang dimiliki oleh properti tersebut [[4](https://www.sciencedirect.com/science/article/pii/S0264275118312241)] [[5](https://www.researchgate.net/publication/333880192_The_price_of_a_view_Estimating_the_impact_of_view_on_house_prices)]. Oleh karena itu, dengan mempertimbangkan variabel-variabel tersebut, yang juga tersedia dalam dataset, kita dapat memperkirakan harga properti dan mengevaluasi seberapa besar pengaruh faktor-faktor tersebut.

Ada banyak algoritma yang dapat dipilih dalam pembuatan model regresi. Salah satu algoritma yang umum digunakan adalah regresi linier. Dengan menggunakan regresi dan menggabungkan faktor-faktor yang relevan dengan properti yang dituju, diharapkan kita dapat memprediksi harga properti yang dimaksud [[6](https://www.researchgate.net/publication/360473275_HOUSE_PRICE_PREDICTION_USING_LINEAR_REGRESSION_IN_ML)] [[7](https://www.researchgate.net/publication/364039805_Prediction_and_Analysis_of_Housing_Price_Based_on_the_Generalized_Linear_Regression_Model)] [[8](https://www.researchgate.net/publication/23534659_Determinants_of_House_Prices_A_Quantile_Regression_Approach)].

## Business Understanding

Pengembangan model prediksi nilai properti memiliki potensi dampak atau manfaat berupa menjadi salah satu alat bantu dalam pengambilan keputusan oleh calon pembeli properti.
Contoh potensi manfaat dari hasil prediksi nilai properti yang akurat adalah membantu pembeli dan penjual membuat keputusan jual ataupu beli yang lebih bijaksana.

### Problem Statements
- Berdasarkan penjelajahan kumpulan data, faktor-faktor apa saja yang memengaruhi dalam menentukan perkiraan nilai properti?
- Bagaimana memproses kumpulan data agar dapat dibuat model prediksi nilai properti?
- Bagaimana cara meningkatkan kinerja model prediksi nilai properti?

### Goals
- Menelusuri semua fitur yang tersedia dalam kumpulan data lalu menganalisis korelasi harga dengan setiap fitur yang ada untuk mengidentifikasi faktor-faktor yang memiliki pengaruh paling signifikan hingga yang paling kurang signifikan terhadap nilai properti
- Melakukan proses pengolahan data dan persiapan data terhadap kumpulan data agar dapat dibuat model prediksi nilai properti
- Melakukan beberapa percobaan dengan berbagai model untuk memilih model terbaik dari sekumpulan model yang telah dibuat untuk prediksi nilai properti


### Solution statements
- Untuk menjelajahi fitur, kami melakukan Analisis Univariat dan Analisis Multivariat. Analisis Univariat digunakan untuk mengeksplorasi data numerik dan data kategorikal. Analisis Multivariat digunakan untuk memahami hubungan antara berbagai fitur. Kami menggunakan 
 teknik seperti catplot, pairplot, dan heatmap untuk mengamati Matrix Korelasi dari fitur-fitur yang ada.
- Untuk mencapai model prediksi yang optimal, kami menjalankan proses Pengelolaan Data yang mencakup Pengumpulan Data, Evaluasi Data, dan Pembersihan Data.
- Untuk menilai kinerja model, kami melakukan pengecekan performa menggunakan metrik evaluasi.

## Data Understanding
Informasi yang digunakan dalam pembuatan model berasal dari data sekunder. Data tersebut diunduh dari platform Kaggle dengan judul dataset yang dikenal sebagai California House Price.

URL: https://www.kaggle.com/datasets/shibumohapatra/house-price

Ini adalah spesifikasi dari kumpulan data yang digunakan untuk pembuatan model:

- Kumpulan data tersedia dalam format CSV.
- Terdapat 20640 catatan dalam kumpulan data dengan 10 fitur yang diukur.
- Kumpulan data terdiri dari 1 entri kategorikal dan 9 entri numerik.
- Terdapat missing value dalam kumpulan data sebanyak 205 catatan.

### Variabel-variabel dalam kumpulan data adalah sebagai berikut:

- longitude : Koordinat geografis yang menunjukkan posisi suatu titik dari arah utara ke selatan pada permukaan bumi (diukur dalam satuan derajat)
- latitude : Koordinat geografis yang menunjukkan posisi suatu titik dari arah timur ke barat pada permukaan bumi (diukur dalam satuan derajat)
- housing_median_age : Usia rata-rata rumah dalam satu blok. Angka yang lebih rendah menunjukkan bangunan yang lebih baru (diukur dalam satuan tahun)
- total_rooms : Jumlah total kamar dalam satu blok (diukur dalam satuan)
- total_bedrooms : Jumlah total kamar tidur dalam satu blok (diukur dalam satuan)
- population : Jumlah total orang yang tinggal dalam satu blok (diukur dalam satuan orang)
- households : Jumlah total rumah tangga dalam satu blok (diukur dalam satuan)
- median_income : Pendapatan median untuk rumah tangga dalam satu blok (diukur dalam Dolar AS)
- ocean_proximity : Posisi relatif lokasi rumah terhadap lautan/laut (kategori 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND')
- median_house_value : Median harga rumah untuk rumah tangga dalam satu blok (diukur dalam Dolar AS)

Untuk memperoleh pemahaman yang lebih mendalam tentang data, dilakukan Analisis Univariat dan Analisis Multivariat, serta Visualisasi Data.

Analisis Univariat merupakan bentuk analisis data yang hanya fokus pada satu variabel. Jenis analisis ini umumnya digunakan untuk memahami distribusi variabel dalam sebuah kumpulan data. Sedangkan, Analisis Multivariat melibatkan lebih dari dua variabel, dan digunakan untuk mengeksplorasi hubungan dan pola dalam data multidimensional.

Selain menggunakan analisis, Visualisasi Data juga penting. Memvisualisasikan data membantu untuk memperoleh pemahaman yang lebih dalam tentang perilaku berbagai fitur dalam kumpulan data.
Pada proyek ini, kami menggunakan berbagai teknik visualisasi, termasuk catplot untuk memvisualisasikan distribusi data kategorikal, pairplot untuk mengeksplorasi hubungan antar fitur, dan heatmap untuk menampilkan korelasi antar fitur dalam bentuk matriks.

Berikut adalah hasil Exploratory Data Analysis (EDA), dimana Gambar 1 menampilkan EDA Analisis Univariat dan Gambar 2 menampilkan EDA Analisis Multivariat.

![download (2)](https://github.com/whiteevl/laporan/blob/main/Screenshot%202024-03-06%20173343.png)

Gambar 1a. Analisis Univariat (Data Kategori)

![download (3)](https://github.com/whiteevl/laporan/blob/main/download.png)

Gambar 1b. Analisis Univariat (Data Numerik)

Berdasarkan Gambar 1a, terlihat bahwa distribusi data kategori untuk 'ocean_proximity' memiliki perbandingan jumlah yang tidak merata. Misalnya, nilai data '<1H OCEAN' mencapai 7607 dengan persentase 43.2%, sementara nilai data 'ISLAND' hanya sebanyak 5. Lebih lanjut, pada Gambar 1b, karakteristik data numerik mencakup:

- Koordinat longitude rumah mayoritas terletak pada rentang -118 hingga -122 derajat, dan koordinat latitude rumah mayoritas berada pada rentang 34 hingga 38 derajat.
- Median usia rumah paling banyak tersebar di rentang 10 hingga 40 tahun, dengan nilai terbanyak terjadi pada usia 50 tahun.
- Rata-rata tertinggi untuk total kamar dan total kamar tidur berada di kisaran 1000-2000 dan 200-400, masing-masing.
- Rata-rata tertinggi untuk populasi dan jumlah rumah tangga terletak di antara 500-1000 dan 200-400.
- Pendapatan median dan nilai rumah median tertinggi masing-masing berada di antara angka 3 dan 200,000.
- Distribusi median nilai rumah condong ke kanan (right-skewed), yang akan memiliki implikasi pada model.

![download (4)](https://github.com/whiteevl/laporan/blob/main/download%20(1).png)

Gambar 2a. Analisis Multivariat (Data Kategori)

![download (5)](https://github.com/whiteevl/laporan/blob/main/download%20(2).png)

Gambar 2b. Analisis Multivariat (Data Numerik)

![download (6)](https://github.com/whiteevl/laporan/blob/main/download%20(3).png)

Gambar 2c. Analisis Multivariat (Correlation Matrix)

Pada Gambar 2a terlihat penyebaran data 'ocean proximity' terhadap 'median house value'. Dari perbandingan rata-rata 'median_house_value' terhadap fitur kategorikal di atas, dapat disimpulkan:

- Fitur 'ocean_proximity' memiliki variasi rata-rata 'median_house_value' yang signifikan, berkisar antara 120,000 hingga 400,000.
- Nilai tertinggi dari 'median_house_value' terdapat pada nilai 'ocean_proximity' 'ISLAND', sementara nilai terendahnya ada pada nilai 'ocean_proximity' 'INLAND'. Oleh karena itu, fitur 'ocean_proximity' memiliki dampak yang penting terhadap rata-rata 'median_house_value'.
- Kesimpulan akhirnya adalah bahwa fitur kategorikal memengaruhi fitur numerik 'median_house_value'.

Pada Gambar 2b, menggunakan fungsi pairplot dari pustaka seaborn, terlihat relasi pasangan dalam kumpulan data. Plot pairplot menunjukkan hubungan antar fitur numerik dalam kumpulan data. Dari pola penyebaran data pada grafik pairplot, terlihat bahwa 'median_income' memiliki korelasi yang signifikan dengan fitur 'median_house_value'. Namun, korelasi antara fitur lainnya terlihat lemah karena tidak membentuk pola yang jelas.

Terakhir, Gambar 2c menunjukkan Matriks Korelasi yang menampilkan hubungan antar fitur dalam bentuk nilai korelasi. Dari hasil observasi, fitur 'median_income' memiliki korelasi yang cukup tinggi (0.63) dengan fitur target 'median_house_value', menunjukkan korelasi yang signifikan antara kedua fitur tersebut. Sementara itu, fitur lainnya memiliki korelasi negatif, yang mengindikasikan bahwa fitur tersebut dapat dihapus dari model.
  
## Data Preparation
Pada tahap Pengolahan Data, dilakukan aktivitas seperti Pengumpulan Data, Evaluasi Data, dan Pembersihan Data.
Pada tahap Pengumpulan Data, data diimpor dengan format yang sesuai agar dapat dibaca dengan baik menggunakan pustaka Pandas.
Untuk tahap Evaluasi Data, beberapa pemeriksaan yang dilakukan meliputi:

- Duplikasi data (data yang identik dengan data lainnya)
- Missing value (data atau informasi yang tidak lengkap atau tidak tersedia)
- Outlier (data yang signifikan berbeda dari nilai rata-rata dalam himpunan data)

Pada tahap Pembersihan Data, secara umum, terdapat tiga metode yang dapat digunakan, yaitu sebagai berikut:

- Dropping (metode yang melibatkan penghapusan sejumlah baris data)
- Imputation (metode yang melibatkan penggantian nilai yang hilang atau tidak tersedia dengan nilai tertentu, seperti median atau mean dari data)
- Interpolation (metode yang menghasilkan titik-titik data baru di antara data yang ada)

Pada kasus proyek ini, tidak ditemukan adanya duplikasi data. Namun, ditemukan Missing Value. Untuk mengatasi masalah ini, metode yang digunakan adalah imputation, di mana nilai yang hilang diganti dengan nilai mean. Sedangkan untuk outlier, metode dropping digunakan dengan menggunakan metode IQR. Nilai IQR sendiri dihitung dengan mengurangkan nilai Q3 dengan Q1 seperti yang dijelaskan berikut ini.

$$ IQR = Q<sub>3</sub> - Q<sub>1</sub> $$

dimana
Q<sub>1</sub> adalah kuartil pertama dan Q<sub>3</sub> adalah kuartil ketiga.

Dengan menggunakan metode IQR, kita dapat menentukan outlier dengan menggunakan nilai batas yang telah ditentukan. Setelah menerapkan metode IQR, jumlah dataset yang awalnya 20640 menjadi 17609. Seluruh proses ini sangat penting dalam upaya menciptakan model yang optimal.

Untuk mengurangi jumlah fitur, dilakukan proses PCA. Teknik reduksi ini merupakan langkah untuk mengurangi dimensi data dengan tetap mempertahankan informasi yang relevan. PCA mengubah data dari ruang dimensi-n ke dalam sistem koordinat baru dengan dimensi m, di mana m lebih kecil dari n. Dalam proyek ini, fitur 'housing_median_age', 'total_rooms', 'total_bedrooms', dan 'households' divisualisasikan untuk mengeksplorasi hubungan antar fitur tersebut, seperti yang terlihat pada Gambar 3 di bawah ini.

![download (7)](https://github.com/whiteevl/laporan/blob/main/download%20(4).png)

Gambar 3 Visualisasi Hubungan antar Fitur sebelum Reduksi PCA

Berdasarkan Gambar 3, dapat diamati bahwa hanya tiga fitur yang memiliki hubungan antara satu sama lain, yaitu 'total_rooms', 'total_bedrooms', dan 'households'. Selanjutnya, kita akan mereduksi ketiga fitur ini menggunakan PCA. Sebelum melakukannya, kita perlu memeriksa proporsi informasi dari ketiga komponen PC tersebut.

```
pca.explained_variance_ratio_.round(3)
```
Hasil kode tersebut menghasilkan keluaran berupa array([0.985, 0.014, 0.001]). Berdasarkan hasil ini, kita dapat menyimpulkan bahwa hanya PC (Principal Component) pertama yang akan dipertahankan. Hal ini disebabkan oleh fakta bahwa 98.5% dari informasi yang terdapat pada ketiga fitur 'total_rooms', 'total_bedrooms', dan 'households' terkandung dalam PC pertama. Sementara itu, hanya 1.4% dan 0.1% informasi yang tersisa terdapat pada PC kedua dan ketiga. Oleh karena itu, PC pertama akan digunakan sebagai representasi fitur 'house properties', menggantikan peran ketiga fitur lainnya ('total_rooms', 'total_bedrooms', 'households').


Setelah proses pembersihan data selesai, kumpulan data dibagi menjadi data latih dan data uji untuk proses Modeling, dengan menggunakan rasio pembagian data 90:10 karena data uji sudah cukup representatif untuk evaluasi model. Detail kumpulan data setelah pembagian adalah sebagai berikut:

- Jumlah sampel dalam kumpulan data latih: 15848
- Jumlah sampel dalam kumpulan data uji: 1761

## Modeling
Seperti yang telah dijelaskan sebelumnya, pilihan model jatuh pada model regresi karena merupakan salah satu algoritma yang sering digunakan dalam pembuatan model prediksi. Secara sederhana, regresi terdiri dari intersep dan kemiringan yang dapat dijelaskan dalam rumus berikut

$$ y = a + bX $$

dimana:

- y merupakan variabel dependen (variabel yang ingin diprediksi)
- a adalah intercept (nilai konstanta yang menunjukkan titik perpotongan garis regresi dengan sumbu Y),
- b adalah slope (koefisien yang mengindikasikan kemiringan garis regresi), dan
- X adalah variabel independen (variabel yang digunakan untuk memprediksi atau menjelaskan variabel dependen dalam model).

Secara umum, regresi digunakan untuk memprediksi pengaruh variabel prediktor terhadap variabel dependen atau untuk menentukan keberadaan hubungan fungsional antara variabel independen (X) dan variabel dependen (y).


Namun, model regresi memiliki kelebihan dan kelemahan, di antaranya:

Kelebihan regresi:

- Mudah digunakan
- Mampu mengidentifikasi kekuatan pengaruh variabel prediktor (variabel independen) terhadap variabel dependen
- Mampu memprediksi nilai atau tren di masa depan

Namun, model regresi memiliki kelemahan karena hasil prediksi dari analisis regresi adalah estimasi, sehingga kemungkinan ketidaksesuaian dengan data aktual tetap ada.

Dalam proyek ini, algoritma regresi yang dibandingkan mencakup regresi linear, regresi ridge, random forest regressor, dan random forest regressor dengan penyetelan hyperparameter. Regresi linear adalah metode analisis data yang memprediksi nilai data yang tidak diketahui dengan menggunakan nilai data yang terkait dan diketahui, yang dimodelkan secara matematis sebagai persamaan linier. Regresi ridge merupakan teknik estimasi koefisien regresi yang melibatkan penambahan konstanta bias c. Random forest adalah sebuah algoritma yang digunakan untuk klasifikasi data dalam skala besar, di mana teknik klasifikasi random forest melibatkan penggabungan pohon melalui pelatihan pada sampel data yang dimiliki.

Untuk meningkatkan kinerja model, dilakukan penyetelan hyperparameter. Beberapa parameter yang disesuaikan meliputi 'n_estimators', 'max_depth', 'min_samples_split', dan 'min_samples_leaf'. Untuk menyederhanakan proses penyetelan, digunakan GridSearchCV. GridSearchCV adalah bagian dari pustaka scikit-learn yang memungkinkan pencarian otomatis nilai hyperparameter. Grid Search adalah metode yang digunakan untuk menemukan parameter optimal yang dapat meningkatkan kinerja model dengan mencoba berbagai kombinasi hyperparameter yang diberikan.

Berikut adalah nilai parameter *tuning*
```
params = {'n_estimators' : [50,80,100],
          'max_depth' : [3,5,10],
           'min_samples_split':[2,3,4],
            'min_samples_leaf': [2,3,4]}
```
Berdasarkan hasil pengujian, terpilih grid.best_params_ yaitu 
```
{'max_depth': 10,
 'min_samples_leaf': 4,
 'min_samples_split': 2,
 'n_estimators': 100}
```
Parameter dengan nilai inilah yang kemudian dibuat sebagai model.

## Evaluation
Adapun metrik yang sebagai alat ukur perfoma model yang dibuat antara lain **MSE · MAE · R<sup>2</sup>**. 

Berikut merupakan rumus dari masing-masing metrik yang digunakan:

$$ MAE = (1/n) Σ |y<sub>i</sub> - ŷ<sub>i</sub>| $$

$$ MSE = (1/n) Σ (y<sub>i</sub> - ŷ<sub>i</sub>)<sup>2</sup> $$

$$ R<sup>2</sup> = 1 - (MSE / Var(y)) $$

y<sub>i</sub> mewakili nilai yang diamati,
ŷ<sub>i</sub> mewakili nilai prediksi,
n adalah jumlah titik data,
Var(y) mewakili varians dari nilai yang diamati.

Berikut adalah penjelasan tentang penggunaan masing-masing metrik yang digunakan:

- MAE (Mean Absolute Error) menghitung rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual. Semakin kecil nilai MAE, semakin baik kualitas model.
- MSE (Mean Squared Error) menghitung rata-rata dari selisih kuadrat antara nilai prediksi dan nilai aktual. Semakin kecil nilai MSE, semakin baik kualitas model.
- R<sup>2</sup> (R-squared) digunakan untuk mengevaluasi seberapa besar variabilitas dalam variabel dependen yang dapat dijelaskan oleh variabel independen.

Tabel 1 berikut merupakan perbandingan 4 buah model yang coba dibandingkan

|     |Model 1|Model 2|Model 3|Model 4|
|---|---|---|---|---|
|R<sup>2</sup>|-7504.309425109431|-7498.0783949337365.146779279|-1.880080745581838|2.432728538383974|
|MSE|8166746.146779279|8163355.360001828|159980.5138016423|174656.3925179131|
|MAE|6509173.648630178|6506404.9997212|133115.5655366269|154364.3230594773|

Tabel 1. Perbandingan Performa MAE, MSE, dan R<sup>2</sup> Model

Berdasarkan data pada Tabel 1, secara umum, Model 3 (RF1) dan Model 4 (RF2) menunjukkan kinerja yang lebih baik, dengan masing-masing memiliki nilai R^2 sebesar -1.880080745581838 dan -2.432728538383974.

Untuk memperdalam perbandingan antara Model 1, 2, 3, dan 4, silakan lihat Gambar 4 di bawah ini.

![download (1)](https://github.com/whiteevl/laporan/blob/main/download%20(5).png)

Gambar 4. Perbandingan Model berdasarkan Nilai Error (dalam 1e6)

Berdasarkan Gambar 4, terlihat bahwa nilai kesalahan pada data latih dan data uji dari Model 3 (RF1) dan Model 4 (RF2) jauh lebih baik daripada model lainnya.

Selain itu, dilakukan perbandingan antara nilai aktual (y_true) dan nilai prediksi harga rumah dari keempat model yang telah dibuat. Tabel 2 berikut menunjukkan hasil evaluasi dari model-model tersebut.


|     |y_true|prediksi_LR|prediksi_RR|prediksi_RF1|prediksi_RF2|
|---|---|---|---|---|---|
|15732|341700|218287.6|218309.3|347466.0|315645.2|

Tabel 2. Perbandingan Model


Dari hasil evaluasi, terlihat bahwa prediksi harga rumah menggunakan Random Forest (RF), baik RF1 (tanpa penyetelan) maupun RF2 (dengan penyetelan), memberikan hasil yang paling mendekati nilai aktual (y_true), di mana nilai y_true adalah 341700, sedangkan nilai prediksi RF1 dan RF2 masing-masing adalah 347466.0 dan 315645.2. Oleh karena itu, dapat disimpulkan bahwa model yang telah dikembangkan mampu memprediksi harga rumah dengan baik menggunakan Random Forest Regressor.


## Referensi:
[1] Hassan, Mohammad Mujaheed & Ahmad, Nobaya & Hariza, Ahmad & Hashim, Ahmad. (2021). Factors Influencing Housing Purchase Decision. 11. 429-443. 10.6007/IJARBSS/v11-i7/10295.

[2] Ariyawansa, Ranthilaka. (2009). Housing Market: A Review of Purchase Decision of Potential Buyers.

[3] Jiang, Jiaxin & Zhang, Jin. (2021). Analysis of County Consumers’ Housing Purchase Intention and Influencing Factors. 

[4] Heymann, Sommervoll. (2019). House prices and relative location

[5] Jayasekare, Ajith & Herath, Shanaka & Wickramasuriya, Rohan & Perez, Pascal. (2019). The price of a view: Estimating the impact of view on house prices. Pacific Rim Property Research Journal. 25. 1-18. 10.1080/14445921.2019.1626543. 

[6] Agarwal, Umang & Gupta, Smriti & Goyal, Madhav. (2022). House Price Prediction using Linear Regression. 10.13140/RG.2.2.11175.62887. 

[7] Li, Xinshu. (2022). Prediction and Analysis of Housing Price Based on the Generalized Linear Regression Model. Computational Intelligence and Neuroscience. 2022. 1-9. 10.1155/2022/3590224. 

[8] Zietz, Joachim & Zietz, Emily & Sirmans, G.. (2008). Determinants of House Prices: A Quantile Regression Approach. The Journal of Real Estate Finance and Economics. 37. 317-333. 10.1007/s11146-007-9053-7. 

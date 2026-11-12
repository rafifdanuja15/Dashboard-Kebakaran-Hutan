# RUN_SUMMARY: Peramalan Hotspot Kecamatan Kalimantan (notebook final, walk-forward expanding)

- Dijalankan: 2026-10-25 13:43:54 | durasi 409.2 menit | LightGBM 4.6.0 | pandas 2.3.3
- Data hotspot: 2000-11-09 s.d. 2026-09-08 | 591 kecamatan | 5 provinsi | origin forecast 2026-09-08
- Skema tunggal: `tanpa_bobot` (Tweedie power 1.3, leaves 63, mcs 80, tanpa pembobotan, 150 fitur)
- Walk-forward expanding: uji 2015-2025 (latih sampai tahun sebelumnya) | holdout 2026 | seed [42, 7, 2026] | mode cepat: False

## 1. Evaluasi per tahun (MAE, RMSE, deviance, PR-AUC, ROC-AUC, bias)

```
           split                       year   dev13     mae    rmse  bias_pct  pr_auc  roc_auc  recall_top20  n_event
0   walk-forward                       2015  1.2143  0.6702  4.7857  -42.8148  0.5991   0.9720        0.2458   6592.0
1   walk-forward                       2016  0.4734  0.1054  0.6018   35.4690  0.1148   0.9711        0.5234    470.0
2   walk-forward                       2017  0.3432  0.0648  0.3294  143.3226  0.0574   0.9770        0.5053     95.0
3   walk-forward                       2018  0.6072  0.1907  1.4787    9.5083  0.2841   0.9756        0.3887   1199.0
4   walk-forward                       2019  0.9390  0.4317  3.8517  -38.0868  0.5388   0.9795        0.3656   3091.0
5   walk-forward                       2020  0.3102  0.0416  0.2500   65.0399  0.0203   0.9596        0.2661    109.0
6   walk-forward                       2021  0.3114  0.0454  0.3604   24.5731  0.0497   0.9805        0.5859    198.0
7   walk-forward                       2022  0.2756  0.0280  0.2398   12.0588  0.0318   0.9564        0.6765     68.0
8   walk-forward                       2023  0.6953  0.2228  1.3297    5.1580  0.3605   0.9734        0.4683   1734.0
9   walk-forward                       2024  0.4729  0.0842  0.6339  -37.7931  0.1574   0.9818        0.4625    493.0
10  walk-forward                       2025  0.4231  0.0870  0.5025   67.8076  0.0616   0.9800        0.4591    281.0
11       holdout                       2026  0.9999  0.4544  3.9398  -49.1373  0.6463   0.9823        0.2485   2604.0
12  walk-forward         2015-2025 gabungan  0.5513  0.1792  1.9780  -22.7714  0.4523   0.9846        0.3398  14330.0
13  walk-forward  2015-2025 rata-rata tahun  0.5514  0.1793  1.3058   22.2039  0.2069   0.9734        0.4497      NaN
```

## 2. Model vs pembanding

```
                               dev13     mae    rmse  bias_pct  pr_auc  roc_auc  recall_top20
split        model                                                                           
holdout      Model            0.9999  0.4544  3.9398  -49.1373  0.6463   0.9823        0.2485
             Persistensi 7h   2.5179  0.4413  3.6144  -32.9904  0.5272   0.8969        0.2270
             Laju 28h         1.9649  0.4431  4.0162  -63.3203  0.5670   0.9461        0.2312
             Prediksi nol    11.7832  0.5225  5.1124 -100.0000     NaN      NaN           NaN
walk-forward Model            0.5513  0.1792  1.9780  -22.7714  0.4523   0.9846        0.3398
             Persistensi 7h   1.2475  0.1935  2.1801    0.0070  0.3305   0.8492        0.2689
             Laju 28h         0.9621  0.2145  2.2846    0.0114  0.2701   0.9141        0.2708
             Prediksi nol     3.6689  0.1632  2.5561 -100.0000     NaN      NaN           NaN
```

## 3. Kecamatan dengan prediksi terbaik yang diplot

Kriteria: dari 15 kecamatan paling aktif per provinsi, dipilih 5 dengan NSE tertinggi (best case, bukan rata-rata). Skill semua kecamatan ada di `tables/kecamatan_skill_semua_*.csv`.

```
    year province district_id                                   kecamatan  aktual_total  prediksi_total    nse  korelasi  bias_pct  dev13
0   2026       61   ID6106061                     Benua Kayong (Ketapang)        1005.0         553.076  0.625     0.866   -44.968  3.904
1   2026       61   ID6106010                      Kendawangan (Ketapang)        1493.0         766.740  0.482     0.824   -48.644  3.463
2   2026       61   ID6110060                         Ella Hilir (Melawi)         873.0         317.224  0.468     0.892   -63.663  2.146
3   2026       61   ID6106072                      Muara Pawan (Ketapang)        1531.0         613.832  0.428     0.802   -59.906  3.876
4   2026       61   ID6110040                          Belimbing (Melawi)         597.0         215.824  0.383     0.812   -63.849  1.753
5   2026       62   ID6208011               Seruyan Hilir Timur (Seruyan)         638.0         348.541  0.706     0.929   -45.370  1.411
6   2026       62   ID6203150                          Mantangai (Kapuas)        3779.0        1674.973  0.520     0.849   -55.677  5.034
7   2026       62   ID6208010                     Seruyan Hilir (Seruyan)        1002.0         428.895  0.513     0.892   -57.196  1.807
8   2026       62   ID6209030                         Kamipang (Katingan)         972.0         310.091  0.484     0.922   -68.098  3.228
9   2026       62   ID6202060  Mentawa Baru/Ketapang (Kotawaringin Timur)         684.0         287.966  0.445     0.765   -57.900  2.505
10  2026       63   ID6305100                   Candi Laras Utara (Tapin)         184.0         104.582  0.539     0.797   -43.162  0.673
11  2026       63   ID6306090          Daha Selatan (Hulu Sungai Selatan)         148.0          61.415  0.476     0.892   -58.504  0.639
12  2026       63   ID6303120                      Simpang Empat (Banjar)         319.0         161.216  0.312     0.592   -49.462  1.816
13  2026       63   ID6305090                 Candi Laras Selatan (Tapin)         126.0          59.185  0.293     0.601   -53.028  0.981
14  2026       63   ID6305060                           Lokpaikat (Tapin)          87.0          32.219  0.290     0.651   -62.967  0.964
15  2026       64   ID6401021                          Batu Engau (Paser)         168.0         131.478  0.642     0.805   -21.739  0.637
16  2026       64   ID6401070                         Muara Komam (Paser)         113.0          51.830  0.526     0.865   -54.133  0.826
17  2026       64   ID6405040                       Pulau Derawan (Berau)        1015.0         441.276  0.518     0.878   -56.525  1.839
18  2026       64   ID6405070                        Gunung Tabur (Berau)         194.0          89.839  0.437     0.796   -53.691  1.966
19  2026       64   ID6405050                          Sambaliung (Berau)         171.0         104.628  0.397     0.686   -38.814  1.609
20  2026       65   ID6502050                    Tanjung Selor (Bulungan)          18.0          10.756  0.496     0.782   -40.244  0.278
21  2026       65   ID6502060              Tanjung Palas Timur (Bulungan)         386.0         162.956  0.337     0.750   -57.783  2.709
22  2026       65   ID6502080              Tanjung Palas Utara (Bulungan)          35.0          11.959  0.085     0.379   -65.830  0.811
23  2026       65   ID6502040                    Tanjung Palas (Bulungan)           7.0           6.622  0.049     0.261    -5.397  0.298
24  2026       65   ID6502030              Tanjung Palas Barat (Bulungan)           7.0          12.321  0.018     0.394    76.011  0.283
25  2023       61   ID6105170                        Tayan Hulu (Sanggau)         294.0         224.721  0.496     0.711   -23.564  1.506
26  2023       61   ID6112070                     Sungai Raya (Kubu Raya)         343.0         245.537  0.381     0.625   -28.415  2.217
27  2023       61   ID6106010                      Kendawangan (Ketapang)         665.0         727.890  0.326     0.573     9.457  2.352
28  2023       61   ID6106020                       Manis Mata (Ketapang)         392.0         225.655  0.293     0.619   -42.435  1.775
29  2023       61   ID6106060              Matan Hilir Selatan (Ketapang)        1680.0        1136.301  0.283     0.542   -32.363  3.772
30  2023       62   ID6208011               Seruyan Hilir Timur (Seruyan)         938.0         829.298  0.667     0.819   -11.589  1.786
31  2023       62   ID6202021           Teluk Sampit (Kotawaringin Timur)         469.0         375.845  0.605     0.802   -19.862  1.086
32  2023       62   ID6204020                Dusun Hilir (Barito Selatan)         903.0         935.647  0.515     0.728     3.615  2.429
33  2023       62   ID6201060                  Kumai (Kotawaringin Barat)         910.0         565.442  0.462     0.736   -37.863  1.626
34  2023       62   ID6210011               Sebangau Kuala (Pulang Pisau)        1708.0        1483.049  0.442     0.666   -13.170  4.224
35  2023       63   ID6306100            Daha Utara (Hulu Sungai Selatan)         224.0         132.461  0.402     0.697   -40.866  1.336
36  2023       63   ID6305100                   Candi Laras Utara (Tapin)         238.0         292.071  0.369     0.638    22.719  1.034
37  2023       63   ID6303120                      Simpang Empat (Banjar)         420.0         529.018  0.314     0.649    25.957  1.057
38  2023       63   ID6306090          Daha Selatan (Hulu Sungai Selatan)         280.0         200.125  0.305     0.567   -28.527  1.522
39  2023       63   ID6308040         Amuntai Selatan (Hulu Sungai Utara)          91.0          54.108  0.274     0.553   -40.540  0.683
40  2023       64   ID6402041                  Siluq Ngurai (Kutai Barat)          84.0          91.486  0.419     0.663     8.912  0.504
41  2023       64   ID6405010                               Kelay (Berau)         126.0          96.783  0.404     0.644   -23.188  0.765
42  2023       64   ID6401021                          Batu Engau (Paser)         189.0         152.045  0.374     0.623   -19.553  1.363
43  2023       64   ID6405040                       Pulau Derawan (Berau)         343.0         209.882  0.273     0.558   -38.810  2.101
44  2023       64   ID6405080                               Segah (Berau)         392.0         229.636  0.236     0.522   -41.419  2.115
45  2023       65   ID6501030                        Kayan Hulu (Malinau)          35.0          42.084  0.113     0.427    20.240  0.411
46  2023       65   ID6502010                             Peso (Bulungan)          21.0          21.406  0.091     0.305     1.933  0.445
47  2023       65   ID6504011                     Krayan Tengah (Nunukan)           7.0           6.870  0.076     0.294    -1.862  0.259
48  2023       65   ID6501090                   Malinau Selatan (Malinau)          14.0           9.217  0.073     0.280   -34.164  0.369
49  2023       65   ID6502040                    Tanjung Palas (Bulungan)          28.0          12.961  0.033     0.209   -53.710  0.685
```

## 4. Forecast 7 hari ke depan

Jendela 2026-09-09 s.d. 2026-09-15; ambang Siaga 1.90 hotspot/7 hari/kecamatan.

```
                    pred_hotspot_7d  aktual_7h_terakhir  kecamatan_siaga  jumlah_kecamatan
provinsi                                                                                  
Kalimantan Tengah             860.9               943.0               65               136
Kalimantan Barat              355.3               571.0               38               168
Kalimantan Timur              166.0               153.0               27                94
Kalimantan Selatan            115.2                77.0               11               146
Kalimantan Utara               21.6                25.0                1                47
```

```
    peringkat district_id                          kecamatan           provinsi  pred_hotspot_7d status
0           1   ID6203150                 Mantangai (Kapuas)  Kalimantan Tengah            69.37  Siaga
1           2   ID6210011      Sebangau Kuala (Pulang Pisau)  Kalimantan Tengah            64.74  Siaga
2           3   ID6208010            Seruyan Hilir (Seruyan)  Kalimantan Tengah            47.97  Siaga
3           4   ID6106060     Matan Hilir Selatan (Ketapang)   Kalimantan Barat            40.79  Siaga
4           5   ID6106010             Kendawangan (Ketapang)   Kalimantan Barat            36.04  Siaga
5           6   ID6210020         Pandih Batu (Pulang Pisau)  Kalimantan Tengah            33.55  Siaga
6           7   ID6202021  Teluk Sampit (Kotawaringin Timur)  Kalimantan Tengah            30.42  Siaga
7           8   ID6209030                Kamipang (Katingan)  Kalimantan Tengah            29.73  Siaga
8           9   ID6271012    Jekan Raya (Kota Palangka Raya)  Kalimantan Tengah            24.65  Siaga
9          10   ID6208011      Seruyan Hilir Timur (Seruyan)  Kalimantan Tengah            24.11  Siaga
10         11   ID6106070       Matan Hilir Utara (Ketapang)   Kalimantan Barat            23.05  Siaga
11         12   ID6202111      Telawang (Kotawaringin Timur)  Kalimantan Tengah            22.51  Siaga
12         13   ID6271020    Bukit Batu (Kota Palangka Raya)  Kalimantan Tengah            21.84  Siaga
13         14   ID6210010       Kahayan Kuala (Pulang Pisau)  Kalimantan Tengah            21.72  Siaga
14         15   ID6206011            Pantai Lunci (Sukamara)  Kalimantan Tengah            19.80  Siaga
```

## 5. Outlook harian 2026-09-09 s.d. 2026-12-31 (114 hari)

Backtest pada tanggal kalender yang sama di tahun lampau (event untuk PR-AUC/ROC-AUC: ada hotspot pada hari itu):

```
              dev13     mae    rmse  bias_pct  pr_auc  roc_auc  korelasi_kal_blok7
model                                                                             
model_400    0.4314  0.0650  0.5681   18.6272  0.1396   0.8792              0.6053
model_600    0.4331  0.0638  0.5680    9.2596  0.1411   0.8809              0.6048
model_200    0.4337  0.0679  0.5700   36.4841  0.1358   0.8762              0.6039
model_100    0.4400  0.0716  0.5685   64.2865  0.1300   0.8702              0.6029
klimatologi  0.5925  0.0965  0.6696  466.0593  0.0835   0.7322              0.5974
persistensi  0.7443  0.1876  0.8067   88.6737  0.0472   0.6370                 NaN
```

Pohon terpilih: 400. Mengalahkan klimatologi (deviance): True.

Total prakiraan per bulan:
```
provinsi  Kalimantan Barat  Kalimantan Selatan  Kalimantan Tengah  Kalimantan Timur  Kalimantan Utara  Kalimantan
bulan                                                                                                            
2026-09             1034.0               481.0             3320.0             519.0              49.0      5402.0
2026-10              519.0               394.0             2247.0             536.0              35.0      3732.0
2026-11               47.0                34.0              209.0              54.0               5.0       348.0
2026-12                4.0                 5.0                7.0              16.0               3.0        34.0
```

## 6. Catatan dan batasan

- Walk-forward: tiap tahun Y dilatih pada seluruh data sampai Y-1 (purge 8 hari); jumlah pohon dari early stopping pada tahun Y-1. Satu skema dipakai untuk semua tahun; skema ini dipilih pada run tuning sebelumnya yang memakai tahun 2019, 2020, dan 2023, sehingga tahun-tahun tersebut tidak sepenuhnya bebas dari seleksi. Hanya holdout 2026 yang murni.
- Target adalah jumlah hotspot 7 hari ke depan, sehingga baris tumpang tindih antar hari; total aktual pada tabel menghitung tiap hotspot hingga 7 kali.
- Prediksi adalah rata-rata bersyarat (Tweedie). MAE paling kecil pada median (hampir selalu 0), sehingga prediktor nol dapat mengalahkan model pada MAE; gunakan deviance, RMSE, PR-AUC, dan ROC-AUC sebagai penilai utama.
- Pada tahun hotspot besar bias cenderung negatif (puncak dikempiskan), pada tahun tenang positif.
- Kecamatan yang diplot dipilih karena prediksinya paling akurat (best case); lihat NSE median kandidat pada catatan run.
- Outlook harian tidak mengenal cuaca masa depan dan menghasilkan kurva halus; andalkan total mingguan/bulanan dan peringkat kecamatan.
- File hotspot hanya berisi baris positif; hari tanpa hotspot dianggap 0.

## 7. Catatan run

- Catatan data: file hotspot hanya berisi baris positif, sehingga hari tanpa hotspot dianggap 0 (tidak dapat dibedakan dari hari tanpa pengamatan).
- Sparsitas: 98.88% sel kecamatan-hari bernilai 0 sejak 2002; hotspot sangat terkonsentrasi pada tahun kering dan beberapa kecamatan.
- Membangun cuaca harian dari GRIB (tp_daily_v3, TP_MODE='interval'); memakan waktu sekitar 15-20 menit ...
- Walk-forward gabungan: deviance 0.5513 (laju 28h 0.9621, persistensi 1.2475); RMSE 1.9780 (laju 28h 2.2846); PR-AUC 0.452; ROC-AUC 0.985; MAE 0.1792 vs prediksi nol 0.1632.
- Holdout 2026: deviance 0.9999 (laju 28h 1.9649); RMSE 3.9398; PR-AUC 0.646; ROC-AUC 0.982; bias -49.1%.
- 2026: NSE median 74 kecamatan aktif = 0.27; 5 terbaik per provinsi memiliki NSE 0.02-0.71.
- 2023: NSE median 75 kecamatan aktif = 0.16; 5 terbaik per provinsi memiliki NSE 0.03-0.67.
- Forecast 2026-09-09 s.d. 2026-09-15 (origin 2026-09-08): total Kalimantan ≈ 1,519 hotspot; 142 kecamatan berstatus Siaga.
- Outlook harian: origin 2026-09-08 -> 2026-12-31 = 114 hari | fitur 145+4 | backtest [2019, 2020, 2023, 2025]
- Outlook: jumlah pohon terbaik 400 | dev13 harian model 0.4314 vs klimatologi 0.5925 (model lebih baik)
- Outlook harian 2026-09-09 s.d. 2026-12-31: total Kalimantan ≈ 9,517 hotspot (model mengalahkan klimatologi pada backtest)
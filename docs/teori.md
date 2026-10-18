# Teori dan pemetaan ke materi P03

Dokumen ini menghubungkan setiap operasi di `imgenhance` dengan slide materi **P03 - Peningkatan Kualitas
Citra (Image Enhancement)**, menjelaskan rumusnya, dan mencatat dua hal pada slide yang perlu
diperhatikan saat mencocokkan hitungan.

## Operasi titik dan look-up table

Pada *operasi titik*, nilai keluaran sebuah piksel hanya bergantung pada nilai masukan piksel itu sendiri:

```
s = T(r)          r, s = 0 ... 255
```

Karena `r` hanya punya 256 kemungkinan nilai, `T` cukup dihitung **satu kali** menjadi tabel 256 isian
(*look-up table*, LUT), lalu dipakai untuk seluruh piksel dengan `cv2.LUT`. Keuntungannya: cepat
(sama saja untuk gambar kecil maupun besar), dan tabel yang sama dapat digambar sebagai kurva `s = T(r)`
di panel perbandingan. Semua operasi titik pada `imgenhance` dibangun dengan cara ini
(`point.py`, `histogram.py`).

Pada gambar berwarna, operasi tonal (negasi, brightness, gamma, ...) diterapkan **per kanal** B, G, R.
Operasi yang bergantung pada sebaran histogram (equalization dan stretching) secara bawaan hanya
mengubah kanal kecerahan Y (ruang YCrCb) supaya perbandingan antar kanal, yaitu warna, tidak bergeser.

## Operasi dan rumusnya

| Operasi | Rumus | Slide | Catatan |
|---|---|---|---|
| RGB ke grayscale | `x = a_r r + a_g g + a_b b`, `a_r + a_g + a_b = 1` | 10 | `average` = (r+g+b)/3 seperti slide; `luminance` = 0,299 / 0,587 / 0,114 (peka mata) |
| Negasi | `s = 255 - r` | 22 | |
| Thresholding | `s = 255` bila `r >= T`, selain itu `0` | 11, 27-29 | `--otsu` memilih `T` otomatis dari histogram |
| Grayscale ke m-bit | `s = langkah * floor(r / langkah)`, `langkah = 256 / 2^m` | 12 | lihat catatan di bawah |
| Brightness | `s = r + t` | 13 | hasil dipotong ke 0-255 |
| Kontras | `s = G (r - P) + P` | 25 | `P` = titik yang tidak berubah |
| Contrast stretching | garis patah `(0,0) - (r1,s1) - (r2,s2) - (255,255)` | 23-26 | mode `minmax`, `percentile`, `custom` |
| Gamma (power-law) | `s = 255 (r/255)^gamma` | 9 | `gamma < 1` mencerahkan, `> 1` menggelapkan |
| Logaritma | `s = c log(1 + r)`, `c = 255 / log 256` | 9 | memperlebar area gelap |
| Histogram equalization | `s_k = round((L-1) sum_{j<=k} n_j / n)` | 14-21 | lihat bagian berikut |
| Gray-level slicing | `s = level` bila `A <= r <= B`, selain itu `0` (atau `r`) | 9 | |
| Bit-plane slicing | `s = 255` bila bit ke-k dari `r` bernilai 1 | 9 | |
| Smoothing, median, sharpening | jendela `k x k` | 7 | pelengkap (operasi aras lokal) |

### Contrast stretching

Tiga mode menentukan titik `(r1, s1)` dan `(r2, s2)`:

- `minmax`: `r1 = r_min`, `r2 = r_max`, `s1 = 0`, `s2 = 255`. Sederhana, tetapi satu piksel pencilan
  sudah cukup menggagalkannya.
- `percentile` (bawaan): `r1` dan `r2` adalah persentil ke-`p` dan ke-`100 - p` histogram (bawaan 1%).
  Piksel pencilan dikorbankan (dipotong) sehingga bagian tengah histogram melebar.
- `custom`: titik belok ditentukan sendiri, misalnya `(70, 30)` dan `(180, 225)` untuk memperkuat
  kontras bagian tengah saja.

### Histogram dan histogram equalization

Histogram `h(r_k) = n_k` menghitung jumlah piksel pada tiap tingkat keabuan; **peluang**
`P(r_k) = n_k / n`. Gambar gelap menumpuk di kiri, gambar terang di kanan, kontras rendah menyempit
di tengah (slide 14-15; dibaca otomatis oleh `analyze`).

Equalization memakai distribusi kumulatif sebagai fungsi transformasi:

```
s_k = (L - 1) * sum_{j=0..k} P(r_j)          dibulatkan ke tingkat terdekat
```

Implementasi (`histogram.equalize_lut`) mengikuti rumus slide persis (tanpa `cdf_min` seperti pada
versi Wikipedia / `cv2.equalizeHist`). Dua perilaku yang disengaja:

- **Citra seragam** (hanya satu tingkat keabuan terisi) dikembalikan apa adanya. Rumus mentah akan
  memetakan semuanya ke 255 (putih), padahal tidak ada kontras yang bisa diratakan.
- Hasil equalization **tidak pernah benar-benar datar**. Operasi titik hanya memindahkan batang
  histogram ke nilai baru (batang yang jatuh di tempat yang sama bergabung), sehingga histogram hasil
  bergerigi dengan celah di antaranya.

## Diagnosis otomatis (`analyze`, `auto`)

`analysis.py` meringkas histogram luminans menjadi rata-rata, simpangan baku, min-maks, **sebaran**
(selisih persentil ke-98 dan ke-2, tahan terhadap piksel pencilan), dan entropi Shannon.

| Kecerahan (rata-rata) | Kontras (sebaran) | Saran |
|---|---|---|
| gelap (< 85) atau terang (> 170) | rendah (< 128) | `equalize` |
| gelap | baik | `gamma` 0,5 |
| terang | baik | `gamma` 2,0 |
| normal | rendah | `stretch` persentil |
| normal | baik | tidak perlu perbaikan |

Ambang (sepertiga rentang untuk kecerahan, separuh rentang untuk sebaran) adalah konstanta di
`analysis.py`; ubah di sana bila materi / dosen memakai batas lain.

## Contoh soal pada materi dan hasil hitung ulangnya

`python examples/soal_materi.py` mencetak semua contoh berikut dengan memakai fungsi pustaka yang sama
dengan aplikasi (dan `tests/test_examples.py` memeriksa angkanya).

**Thresholding (slide 29)** - T = 128: piksel `< 128` menjadi 0, `>= 128` menjadi 255. Perhatikan nilai
tepat `128` pada baris ketiga: hasilnya 255.

**Kontras G = 2, P = 10 (slide 26)** - mis. `15 -> 2(15 - 10) + 10 = 20`, `50 -> 90`, `20 -> 30`.

**Histogram equalization L = 8, 64 x 64 piksel (slide 17-21; tabel hasil di slide 20)**:

| k | r_k | n_k | P(r_k) | kumulatif | s_k (dibulatkan ke 8 tingkat) |
|---|---|---|---|---|---|
| 0 | 0,00 | 790 | 0,19 | 0,19 | 0,14 |
| 1 | 0,14 | 1023 | 0,25 | 0,44 | 0,43 |
| 2 | 0,29 | 850 | 0,21 | 0,65 | 0,71 |
| 3 | 0,43 | 656 | 0,16 | 0,81 | 0,86 |
| 4 | 0,57 | 329 | 0,08 | 0,89 | 0,86 |
| 5 | 0,71 | 245 | 0,06 | 0,95 | 1,00 |
| 6 | 0,86 | 122 | 0,03 | 0,98 | 1,00 |
| 7 | 1,00 | 81 | 0,02 | 1,00 | 1,00 |

Piksel dengan nilai baru yang sama digabung: `s = 0,86` menampung `656 + 329 = 985` piksel,
`s = 1,00` menampung `245 + 122 + 81 = 448` piksel.

## Catatan pada slide

1. **Slide 12, "4 bit (0-64)".** Contoh `X = 100` dengan langkah 64 menghasilkan `X baru = 64`. Langkah 64
   berarti `256 / 64 = 4` tingkat, yaitu **2 bit** (`2^2 = 4`), bukan 4 bit. Dengan 4 bit sungguhan
   (16 tingkat, langkah 16) hasilnya `96`. Aplikasi memakai rumus yang benar: opsi `--bits m` selalu
   berarti `2^m` tingkat. Opsi `--rescale` menyebar tingkat hasil merata sampai 255 supaya putih tetap
   putih.
2. **Slide 20, tabel jumlah piksel hasil transformasi.** Ada dua angka yang tidak cocok dengan hitungannya:
   - Baris `s = 1,00` tertulis `254 + 122 + 81 = 448`. Jumlah 254 + 122 + 81 sebenarnya 457; yang
     benar `245 + 122 + 81 = 448`, sesuai `n_5 = 245` pada tabel histogram awal. (Total seluruh
     piksel hanya 4096 bila `n_5 = 245`.)
   - Baris `s = 0,86` tertulis `P = 0,23`. Jumlahnya 985 dari 4096 piksel = `0,2405`, yaitu **0,24**.
     Dengan 0,24 seluruh peluang berjumlah 1,00 (dengan 0,23 hanya 0,99).

   Bila jawaban tugas harus persis seperti slide, tuliskan angka slide dan sebutkan hitungan sebenarnya.

## Kapan memakai operasi yang mana

| Gejala | Histogram | Coba |
|---|---|---|
| Terlalu gelap, detail masih ada | menumpuk di kiri, rentang lebar | `gamma` < 1, `log`, atau `brightness` |
| Terlalu terang | menumpuk di kanan | `gamma` > 1 |
| Pudar / berkabut | sempit di tengah | `stretch` (persentil), `contrast`, atau `equalize` |
| Gelap *dan* sempit | menumpuk di satu sisi, sempit | `equalize` |
| Ingin memisahkan objek dari latar | dua puncak | `threshold --otsu` |
| Ingin melihat struktur bit | - | `bitplane --bit 7 ... 0`, `mbit` |
| Bintik garam-merica | - | `median` |

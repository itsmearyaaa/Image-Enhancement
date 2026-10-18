# Gambar contoh

Lima gambar di folder ini dibuat oleh [`../generate_samples.py`](../generate_samples.py) dari foto bawaan
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html) (`skimage.data`), yang
dilisensikan bebas. Tiga di antaranya **sengaja dirusak** supaya masalah yang dibahas pada materi (terlalu
gelap, kontras rendah, terlalu terang) terlihat jelas pada histogramnya.

| File | Sumber | Perlakuan | Rata-rata keabuan | Saran `analyze` |
|---|---|---|---|---|
| `sample_astronaut_gelap.png` | `data.astronaut()` | digelapkan (gamma besar + redup) | 48 | gelap: `gamma` 0,5 |
| `sample_chelsea_pudar.png` | `data.chelsea()` | kontras dipadatkan ke sekitar abu-abu tengah | 126 | kontras rendah: `stretch` |
| `sample_coffee_terang.png` | `data.coffee()` | dicerahkan sampai banyak piksel terpotong di 255 | 188 | terang: `gamma` 2,0 |
| `sample_camera.png` | `data.camera()` | tanpa perubahan (grayscale) | 129 | sudah baik |
| `sample_coins.png` | `data.coins()` | tanpa perubahan (grayscale) | 97 | sudah baik |

## Atribusi dan lisensi foto asal

| Foto | Keterangan (menurut dokumentasi `skimage.data`) |
|---|---|
| `astronaut` | Foto Eileen Collins, astronaut NASA; dari *NASA Great Images database*; tidak ada pembatasan hak cipta yang diketahui, domain publik. |
| `chelsea` | Kucing Chelsea; CC0 oleh fotografer Stefan van der Walt. |
| `coffee` | Foto atas izin Pikolo Espresso Bar; CC0 oleh fotografer Rachel Michetti. |
| `camera` | *Cameraman*; CC0 oleh fotografer Lav Varshney. |
| `coins` | Koin Yunani dari Pompeii; dari koleksi terbuka Brooklyn Museum; tidak ada pembatasan hak cipta yang diketahui. |

Informasi di atas disalin dari docstring `scikit-image` 0.26; periksa dokumentasi resminya bila gambar
akan dipublikasikan ulang. Lisensi kode project ini (MIT) tidak berlaku bagi foto-foto tersebut.

## Membuat ulang

```bash
pip install scikit-image
python examples/generate_samples.py
```

File hasilnya sudah disertakan, jadi langkah ini tidak diperlukan untuk memakai aplikasi.

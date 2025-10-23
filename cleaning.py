import pandas as pd

df = pd.read_csv('AnalisisKepuasanJurusan.csv')
df.info()

#HAPUS KOLOM YANG TIDAK DIBUTUHKAN
df = df.drop(columns = [
    'Timestamp',
    'Apakah Anda bersedia untuk mengisi pertanyaan-pertanyaan berikut ini?', 
    'No. WhatsApp\nContoh : 087778669888', 
    'Column 19']
)

#BERSIHKAN NAMA KOLOM
df.columns = df.columns.str.replace(r"\n.*", "", regex=True)  # hapus teks setelah newline (\n)
df.columns = df.columns.str.replace(r"Contoh.*", "", regex=True)  # hapus kata 'Contoh'
df.columns = df.columns.str.strip()  # hapus spasi di awal/akhir

kolom_kategori = [
    'Fakultas',
    'Program Studi',
    'Dari mana Anda pertama kali mengetahui informasi tentang jurusan ini?',
    'Apa alasan utama Anda memilih jurusan ini?',
    'Apakah Anda pernah ingin pindah jurusan?',
    'Seberapa relevan kurikulum jurusan Anda dengan kebutuhan dunia kerja?',
    'Bagaimana tingkat kesesuaian jurusan yang Anda pilih dengan minat Anda?',
    'Bagaimana penilaian Anda terhadap prospek kerja lulusan dari jurusan ini?'
]

for col in kolom_kategori:
    df[col] = df[col].astype(str).str.strip().str.title()

#UBAH KOLOM NUMERIK MENJADI TIPE NUMERIK
kolom_numerik = [
    'Secara keseluruhan, bagaimana tingkat kepuasan Anda terhadap jurusan yang Anda pilih?',
    'Seberapa sulit mata kuliah yang ada di jurusan Anda?',
    'Seberapa tinggi motivasi Anda mengikuti perkuliahan di jurusan ini?',
    'Berapa banyak mata kuliah di jurusan ini yang menurut Anda benar-benar sesuai dengan minat Anda?',
    'Berapa kali dalam satu minggu Anda merasa stress dan pusing karena tekanan tugas dari jurusan yang Anda pilih?',
    'Dari total mata kuliah yang Anda tempuh, berapa banyak yang menurut Anda bermanfaat secara langsung untuk persiapan karier Anda?'
]

for col in kolom_numerik:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#MAPPING
df['Program Studi'] = df['Program Studi'].replace({
    'sains data' : 'Sains Data',
    'AKUNTANSI' : 'Akuntansi',
    'Dkv' : 'Desain Komunikasi Visual',
    'dkv' : 'Desain Komunikasi Visual',
    'hukum' : 'Hukum',
    'Arsitektur 93’' : 'Arsitektur',
    'fisika' : 'Fisika',
    'agroteknologi' : 'Agroteknologi'
})

#SAVE FILE
df.to_csv('AnalaisisKepuasan_cleaned.csv', index=False)
print("\n✅ Data cleaned berhasil disimpan sebagai 'data_cleaned.csv'")
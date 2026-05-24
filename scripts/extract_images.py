import json, os, base64
nb_path = r"c:\Users\Acer\Downloads\eda_notebook.ipynb"
out_dir = r"c:\Users\Acer\OneDrive\Desktop\Byte2Beat\outputs"
os.makedirs(out_dir, exist_ok=True)
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
count = 0
for c_idx, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') != 'code':
        continue
    outputs = cell.get('outputs', [])
    for o_idx, out in enumerate(outputs):
        data = out.get('data', {})
        for mime in ('image/png', 'image/jpeg'):
            if mime in data:
                img_b64 = data[mime]
                if isinstance(img_b64, list):
                    img_b64 = ''.join(img_b64)
                img = base64.b64decode(img_b64)
                ext = 'png' if mime=='image/png' else 'jpg'
                fname = f"fig_cell{c_idx+1}_{o_idx+1}.{ext}"
                path = os.path.join(out_dir, fname)
                with open(path, 'wb') as wf:
                    wf.write(img)
                print('Saved', path)
                count += 1
print(f'Saved {count} images to {out_dir}')

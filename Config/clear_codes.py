import json

ruta = r"C:\Users\heoctor\Desktop\WANT\BANREP\Translete-Update\Config\validate.json"

with open(ruta, 'r', encoding='utf-8') as f:
    data = json.load(f)

modificados = 0

for message in data:
    msg_type = message.get('messageType', '').lower()
    
    if msg_type.startswith('camt') or msg_type.startswith('pacs'):
        if 'fields' in message:
            for field in message['fields']:
                field_name = field.get('field', '').upper()
                
                if 'BICFI' in field_name or 'ANYBIC' in field_name:
                    if 'valueDbUser' not in field:
                        field['valueDbUser'] = 'BICFI'
                        modificados += 1

print(f"Campos modificados: {modificados}")

salida = ruta.replace('.json', '_con_bicfi.json')

with open(salida, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Listo: {salida}")
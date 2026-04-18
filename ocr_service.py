"""
Servicio OCR standalone usando Gemini Vision.

Envía PDFs/imágenes directamente a Gemini 2.5 Flash que extrae
todos los campos de factura en una sola llamada de visión.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@dataclass
class FacturaExtraida:
    """Factura extraída y parseada."""
    proveedor: str = ""
    numero_factura: str = ""
    fecha: str = ""
    importe_total: str = ""
    importe_base: str = ""
    iva: str = ""
    tipo_iva: str = ""
    nif_proveedor: str = ""
    nif_cliente: str = ""
    descripcion: str = ""
    pasajeros: str = ""
    num_pasajeros: str = ""
    localizador: str = ""
    numero_expediente: str = ""
    fecha_inicio: str = ""
    fecha_fin: str = ""
    moneda: str = ""
    notas: str = ""
    ai_extracted: bool = False

    def to_dict(self) -> dict:
        return {
            "proveedor": self.proveedor,
            "numero_factura": self.numero_factura,
            "fecha": self.fecha,
            "importe_total": self.importe_total,
            "importe_base": self.importe_base,
            "iva": self.iva,
            "tipo_iva": self.tipo_iva,
            "nif_proveedor": self.nif_proveedor,
            "nif_cliente": self.nif_cliente,
            "descripcion": self.descripcion,
            "pasajeros": self.pasajeros,
            "num_pasajeros": self.num_pasajeros,
            "localizador": self.localizador,
            "numero_expediente": self.numero_expediente,
            "fecha_inicio": self.fecha_inicio,
            "fecha_fin": self.fecha_fin,
            "moneda": self.moneda,
            "notas": self.notas,
            "ai_extracted": self.ai_extracted,
        }


def _clean(val) -> str:
    s = str(val).strip() if val is not None else ""
    if s.lower() in ("none", "null", "n/a", "nan", "undefined"):
        return ""
    return s


def _clean_num(val) -> str:
    s = _clean(val)
    if not s:
        return ""
    s = s.replace(",", ".")
    try:
        return str(float(s))
    except (ValueError, TypeError):
        return s


def _dict_to_factura(data: dict) -> FacturaExtraida:
    importe_total = _clean_num(data.get("importe_total"))
    importe_base = _clean_num(data.get("importe_base"))
    iva = _clean_num(data.get("iva"))

    if not importe_base and importe_total:
        importe_base = importe_total
    if not iva and importe_total:
        iva = "0.0"

    return FacturaExtraida(
        proveedor=_clean(data.get("proveedor")),
        numero_factura=_clean(data.get("numero_factura")),
        fecha=_clean(data.get("fecha")),
        importe_total=importe_total,
        importe_base=importe_base,
        iva=iva,
        tipo_iva=_clean(data.get("tipo_iva")),
        nif_proveedor=_clean(data.get("nif_proveedor")),
        nif_cliente=_clean(data.get("nif_cliente")),
        descripcion=_clean(data.get("descripcion")),
        pasajeros=_clean(data.get("pasajeros")),
        num_pasajeros=_clean(data.get("num_pasajeros")),
        localizador=_clean(data.get("localizador")),
        numero_expediente=_clean(data.get("numero_expediente")),
        fecha_inicio=_clean(data.get("fecha_inicio")),
        fecha_fin=_clean(data.get("fecha_fin")),
        moneda=_clean(data.get("moneda")),
        notas=_clean(data.get("notas")),
        ai_extracted=True,
    )


# Default columns for invoice extraction
DEFAULT_COLUMNS = [
    "proveedor", "nif_proveedor", "numero_factura", "fecha",
    "importe_total", "importe_base", "iva", "tipo_iva",
    "descripcion", "pasajeros", "num_pasajeros",
    "localizador", "nuestro_localizador",
    "fecha_inicio", "fecha_fin", "moneda", "notas",
]

# Human-readable labels
COLUMN_LABELS = {
    "proveedor": "Proveedor",
    "nif_proveedor": "NIF Proveedor",
    "numero_factura": "N. Factura",
    "numero_expediente": "N. Expediente",
    "fecha": "Fecha Emision",
    "importe_total": "Total",
    "importe_base": "Base Imponible",
    "iva": "IVA",
    "tipo_iva": "Tipo IVA",
    "nif_cliente": "NIF Cliente",
    "descripcion": "Descripcion",
    "pasajeros": "Pasajeros",
    "num_pasajeros": "N. Pasajeros",
    "localizador": "Localizador",
    "nuestro_localizador": "Nuestro Localizador",
    "fecha_inicio": "F. Inicio",
    "fecha_fin": "F. Fin",
    "moneda": "Moneda",
    "notas": "Notas",
}


class OCRService:
    """Servicio OCR con Gemini Vision."""

    def __init__(self, gemini_api_key: str = ""):
        self._gemini_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        self._gemini_client = None

        if self._gemini_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self._gemini_key)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._gemini_client is not None

    def extract(self, file_path: Path, custom_prompt: str = "") -> list[FacturaExtraida]:
        """
        Extrae facturas de un PDF/imagen usando Gemini Vision.

        Args:
            file_path: Ruta al archivo PDF/JPG/PNG.
            custom_prompt: Prompt adicional del usuario para personalizar la extraccion.

        Returns:
            Lista de FacturaExtraida.
        """
        if not self._gemini_client:
            return []

        ext = file_path.suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return []

        try:
            uploaded = self._gemini_client.files.upload(file=str(file_path))

            prompt = self._build_prompt(custom_prompt)

            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, uploaded],
            )

            response_text = response.text.strip()

            # Clean markdown wrapper
            if response_text.startswith("```"):
                response_text = re.sub(r'^```\w*\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)

            data_list = json.loads(response_text)

            if not isinstance(data_list, list):
                data_list = [data_list]

            facturas = []
            for data in data_list:
                fac = _dict_to_factura(data)
                if fac and (fac.numero_factura or fac.importe_total):
                    facturas.append(fac)

            # Track API usage
            try:
                from usage_tracker import log_usage
                um = response.usage_metadata
                if um:
                    log_usage(
                        operation="extract",
                        filename=file_path.name,
                        prompt_tokens=um.prompt_token_count or 0,
                        output_tokens=um.candidates_token_count or 0,
                        thinking_tokens=getattr(um, "thoughts_token_count", 0) or 0,
                        total_tokens=um.total_token_count or 0,
                        num_facturas=len(facturas),
                    )
            except Exception:
                pass

            # Clean uploaded file
            try:
                self._gemini_client.files.delete(name=uploaded.name)
            except Exception:
                pass

            return facturas

        except Exception as e:
            raise RuntimeError(f"Error al procesar con Gemini Vision: {str(e)}")

    def customize_columns(self, user_request: str) -> dict:
        """
        Usa Gemini para interpretar lo que el usuario necesita y devolver
        la configuracion de columnas.

        Args:
            user_request: Descripcion en lenguaje natural de lo que necesita.

        Returns:
            Dict con columns (list), prompt_extra (str), explanation (str).
        """
        if not self._gemini_client:
            return {"columns": DEFAULT_COLUMNS, "prompt_extra": "", "explanation": "Gemini no disponible."}

        system_prompt = f"""Eres un asistente que ayuda a configurar la extraccion de datos de facturas.
El usuario te dira que campos necesita extraer de sus facturas/documentos.

Campos disponibles por defecto:
{json.dumps(COLUMN_LABELS, ensure_ascii=False, indent=2)}

Tu tarea:
1. Entender que columnas necesita el usuario
2. Si pide campos que no estan en la lista, crea nuevos nombres de campo (snake_case)
3. Genera un fragmento de prompt adicional para mejorar la extraccion de esos campos

Responde SOLO con JSON valido:
{{
  "columns": ["campo1", "campo2", ...],
  "column_labels": {{"campo1": "Etiqueta 1", "campo2": "Etiqueta 2", ...}},
  "prompt_extra": "Instrucciones adicionales para el modelo de extraccion...",
  "explanation": "Explicacion breve para el usuario de lo que se ha configurado"
}}"""

        try:
            response = self._gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=system_prompt + "\n\nPeticion del usuario: " + user_request,
            )

            # Track API usage
            try:
                from usage_tracker import log_usage
                um = response.usage_metadata
                if um:
                    log_usage(
                        operation="customize_columns",
                        filename="",
                        prompt_tokens=um.prompt_token_count or 0,
                        output_tokens=um.candidates_token_count or 0,
                        thinking_tokens=getattr(um, "thoughts_token_count", 0) or 0,
                        total_tokens=um.total_token_count or 0,
                    )
            except Exception:
                pass

            text = response.text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            return json.loads(text)

        except Exception:
            return {
                "columns": DEFAULT_COLUMNS,
                "column_labels": COLUMN_LABELS,
                "prompt_extra": "",
                "explanation": "No se pudo interpretar la peticion. Se usan los campos por defecto.",
            }

    def _build_prompt(self, custom_prompt: str = "") -> str:
        extra = ""
        if custom_prompt:
            extra = f"\n\nINSTRUCCIONES ADICIONALES DEL USUARIO:\n{custom_prompt}\n"

        return f"""Eres un experto en extraccion de datos de facturas y documentos.
Analiza este documento. Puede contener UNA o MULTIPLES facturas.
Paginas con solo codigos de barras NO son facturas (ignoralas).

Para CADA factura encontrada, extrae TODOS estos campos:
- proveedor: empresa que EMITE la factura (NO el cliente). Nombre completo.
- nif_proveedor: NIF/CIF del emisor. BUSCALO EN TODO EL DOCUMENTO: puede estar en texto
  vertical en los margenes laterales, en el pie de pagina, junto a "con NIF:", "CIF:",
  "N.I.F.", en la cabecera, CUALQUIER LUGAR. Formato: letra + 8 digitos (ej: B65048290).
- numero_factura: numero completo de la factura (ej: 0000/G2200016, 00009151/SE22)
- numero_expediente: numero de expediente si aparece
- localizador: codigo de reserva/localizador (DISTINTO del n factura y expediente)
- fecha: fecha de EMISION de la factura en DD/MM/YYYY
- fecha_inicio: fecha de entrada/check-in en DD/MM/YYYY
- fecha_fin: fecha de salida/check-out en DD/MM/YYYY
- importe_total: importe TOTAL final (lo que se paga). Numero con decimales.
- importe_base: base imponible antes de impuestos. Si no hay impuestos, igual al total.
- iva: importe del IVA. Si exento/no sujeto, poner 0.0
- tipo_iva: "exento", "no_sujeto", "0%", "10%", "21%"
- nif_cliente: NIF/CIF del CLIENTE
- descripcion: descripcion del servicio, hotel/alojamiento + destino
- pasajeros: TODOS los nombres de pasajeros/viajeros separados por |
- num_pasajeros: numero total de pasajeros
- moneda: EUR por defecto
- notas: observaciones relevantes
{extra}
Responde SOLO con un JSON array valido, sin texto adicional ni bloques markdown:
[{{"proveedor":"...","nif_proveedor":"...","numero_factura":"...","numero_expediente":"...","localizador":"...","fecha":"...","fecha_inicio":"...","fecha_fin":"...","importe_total":"...","importe_base":"...","iva":"...","tipo_iva":"...","nif_cliente":"...","descripcion":"...","pasajeros":"...","num_pasajeros":"...","moneda":"EUR","notas":"..."}}]

Si hay multiples facturas, devuelve un elemento por factura en el array."""


def extract_date_from_filename(filename: str) -> str:
    """Intenta extraer una fecha del nombre del archivo."""
    m = re.search(r'(\d{2})[-_](\d{2})[-_](\d{4})', filename)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    m = re.search(r'(\d{4})[-_](\d{2})[-_](\d{2})', filename)
    if m:
        return f"{m.group(3)}/{m.group(2)}/{m.group(1)}"
    return ""

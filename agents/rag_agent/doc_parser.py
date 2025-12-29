import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional # أضفنا Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode, 
    RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureItem, TableItem

class MedicalDocParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Medical Document Parser initialized!")

    def parse_document(
            self,
            document_path: str,
            output_dir: str,
            image_resolution_scale: float = 1.0,
            do_ocr: bool = True,
            do_tables: bool = True,
            do_formulas: bool = True,
            do_picture_desc: bool = False
        ) -> Tuple[Any, List[str]]:
        
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        pipeline_options = PdfPipelineOptions(
            generate_page_images=True,
            generate_picture_images=True,
            images_scale=image_resolution_scale,
            do_ocr=do_ocr,
            do_table_structure=do_tables,
            do_formula_enrichment=do_formulas,
            do_picture_description=do_picture_desc
        )
        
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        conversion_res = converter.convert(document_path)
        doc_filename = conversion_res.input.file.stem
        
     
        for page_no, page in conversion_res.document.pages.items():
            # التحقق من أن الصفحة وصورتها موجودتان قبل الحفظ
            if page and page.image and page.image.pil_image:
                page_image_filename = output_dir_path / f"{doc_filename}-{page_no}.png"
                with page_image_filename.open("wb") as fp:
                    page.image.pil_image.save(fp, format="PNG")
            else:
                self.logger.warning(f"Page {page_no} has no image data.")

        table_counter = 0
        picture_counter = 0
        image_paths = []
        
        for element, _level in conversion_res.document.iterate_items():
            
            if isinstance(element, TableItem):
                table_counter += 1
                try:
                    img = element.get_image(conversion_res.document)
                    if img: 
                        element_image_filename = output_dir_path / f"{doc_filename}-table-{table_counter}.png"
                        img.save(element_image_filename, "PNG")
                except Exception as e:
                    self.logger.error(f"Failed to save table image: {e}")
            
            
            if isinstance(element, PictureItem):
                picture_counter += 1
                try:
                    img = element.get_image(conversion_res.document)
                    if img: 
                        picture_path = f"{doc_filename}-picture-{picture_counter}.png"
                        element_image_filename = output_dir_path / picture_path
                        img.save(element_image_filename, "PNG")
                        image_paths.append(str(element_image_filename))
                except Exception as e:
                    self.logger.error(f"Failed to save picture image: {e}")
        
        
        images = []
        if hasattr(conversion_res.document, 'pictures'):
            for picture in conversion_res.document.pictures:
                # التحقق من أن الكائن له مرجع وصورة
                if picture and picture.image and picture.image.uri:
                    images.append(str(picture.image.uri))
        
        return conversion_res.document, images
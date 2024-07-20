from string import Template

class PathConfig(object):
    def __init__(self, brain_id, section_id) -> None:
        self.brain_id = brain_id
        self.section_id = section_id

        self.gjson_dir_path = Template('/storage/BrainSAM/data/geojson/$brain_id')
        self.img_dir_path = Template('/storage/BrainSAM/data/img/highres/$brain_id')
        self.zarr_dir_path = Template('zarr_n5/optimum_1024/$brain_id')
        self.metada_dir_path = Template('/storage/BrainSAM/data/metadata/$brain_id')

        self.gjson_path_template = Template('/storage/BrainSAM/data/geojson/$brain_id/$section_id.geojson')
        self.img_path_template = Template('/storage/BrainSAM/data/img/highres/$brain_id/$section_id.jp2')
        self.metadata_path_template = Template('/storage/BrainSAM/data/metadata/$brain_id/$section_id.json')
        self.zarr_store_path_template = Template('zarr_n5/optimum_1024/$brain_id/$section_id.n5')
    
    def gjson_dir(self):
        return self.gjson_dir_path.substitute(brain_id=self.brain_id)
    
    def img_dir(self):
        return self.img_dir_path.substitute(brain_id=self.brain_id)
    
    def metadata_dir(self):
        return self.metada_dir_path.substitute(brain_id=self.brain_id)
    
    def zarr_dir(self):
        return self.zarr_dir_path.substitute(brain_id=self.brain_id)
    
    def gjson_path(self):
        return self.gjson_path_template.substitute(brain_id=self.brain_id, section_id=self.section_id)

    def img_path(self):
        return self.img_path_template.substitute(brain_id=self.brain_id, section_id=self.section_id)

    def metadata_path(self):
        return self.metadata_path_template.substitute(brain_id=self.brain_id, section_id=self.section_id)
    
    def zarr_store_path(self):
        return self.zarr_store_path_template.substitute(brain_id=self.brain_id, section_id=self.section_id)
    
from lib.patch_db import PatchDB
from lib.path_config import PathConfig
from lib.sync.registrar import MetaSync
import concurrent.futures
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")

import os

def get_tasks(brain_id):
    brain_ids = [141,142,244,222]
    brain_sections = {b:[] for b in brain_ids}
    task_sections = []
    for brain_id in brain_ids:
        path_config = PathConfig(brain_id, None)
        brain_sections[brain_id] = [int(section.replace('.geojson','')) for section in os.listdir(path_config.gjson_dir())]
        for section_id in brain_sections[brain_id]:
            task_sections.append((brain_id, section_id))
    return task_sections


def work_unit(brain_id, section_id, patch_size, stride):
    try:
        pdb = PatchDB(brain_id, section_id, patch_size, stride)
        pdb.populate_db()
        pdb.qc_check()
        # registrar.sync_patches(pdb)
        return pdb
    except Exception as e:
        with open('error_log.txt','a') as f:
            f.write(f'BrainID:{brain_id},SectionID:{section_id},Error:{e}\n')    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--brain_id', type=int)
    args = parser.parse_args()
    task_sections = get_tasks(args.brain_id)
    registrar = MetaSync()
    errors = 0
    with tqdm(total=len(task_sections)) as pbar:
        print(f'Processing {args.brain_id}')
        pdbs = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
            futures = []
            for brain_id, section_id in task_sections:
                f = executor.submit(work_unit, brain_id, section_id, 1024, 512)
                futures.append(f)
            for future in concurrent.futures.as_completed(futures):
                pdb = future.result()
                pdbs.append(pdb)
                if pdb:
                    registrar.sync_patches(pdb)
                else:
                    errors += 1
                pbar.update(1)
        # print(f'Processing {args.brain_id} - Syncing')
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = []
        #     for pdb in pdbs:
        #         if pdb:
        #             f = executor.submit(registrar.sync_patches, pdb)
        #             futures.append(f)
        #         else:
        #             errors += 1
        #     for future in concurrent.futures.as_completed(futures):
        #         pbar.update(1)
            # for f in results:
            #     if not f.result():
            #         errors += 1
            print(f'Errors: {errors}')
import argparse
import csv
import json
import os
import tarfile
from collections import OrderedDict
# import lang.py.analytic_pb2 as pb

from pprint import pprint

status_map = {
        "OPT_OUT_NONE": "Processed",
        "OPT_OUT_ALL": "OptOutAll",
        "OPT_OUT_DETECTION": "OptOutDetection",
        "OPT_OUT_LOCALIZATION": "OptOutLocalization"
        }

def getID(filename):
    path, ext = os.path.splitext(filename)
    return path.split("/")[-1]

def get_mdl_mask_name(data):
    try:
        prefix="mask"
        mask_uri = data["imgManip"]["localization"]["mask"]["uri"]
        # TODO may need to do some sort of file check?  NIST expects png, but may be best to abort here if different format?
        mask_name = getID(data['imgManipReq']["image"]["uri"])+".png"
        return os.path.join(prefix,mask_name)
    except:
        return ""

def get_vdl_mask_names(data):
    prefix="mask"
    video_frame_segments = []
    audio_sample_segments = []
    video_frame_optout_segments = []
    if "localization" in data["vidManip"]:
        if "frameDetection" in data["vidManip"]["localization"]:
            print("Adding Frame Detection")
            for fd in data["vidManip"]["localization"]["frameDetection"]:
                try:
                    interval = [fd["range"]["start"], fd["range"]["end"]]
                    video_frame_segments.append(interval)
                except Exception as e:
                    print("Error parsing frameDetection: {!s}".fmt(e))
        if "frameOptout" in data["vidManip"]["localization"]:
            print("Adding Frame OptOut")
            for opt in data["vidManip"]["localization"]["frameOptout"]:
                try:
                    interval = [opt["start"],opt["end"]]
                    video_frame_optout_segments.append(interval)
                except Exception as e:
                    print("Error parsing frameOptOut: {!s}".fmt(e))
        if "audioDetection" in data["vidManip"]["localization"]:
            print("Adding Audio Detection")
            for ad in data["vidManip"]["localization"]["audioDetection"]:
                try:
                    interval = [ad["range"]["start"],ad["range"]["end"]]
                    audio_sample_segments.append(interval)
                except Exception as e:
                    print("Error parsing audioDetection: {!s}".fmt(e))
    else:
        print("No localization data in results")
    return video_frame_segments, audio_sample_segments, video_frame_optout_segments

def get_sdl_mask_names(data):
    try:
        prefix="mask"
        from_mask_uri = data["imgSplice"]["link"]["fromMask"]["mask"]["uri"]
        from_mask_name = data['imgSpliceReq']["probeImage"]["uri"].split("/")[-1].split(".")[0]+"_from_mask.png"
        to_mask_uri = data["imgSplice"]["link"]["toMask"]["mask"]["uri"]
        to_mask_name = data['imgSpliceReq']["donorImage"]["uri"].split("/")[-1].split(".")[0]+"_to_mask.png"
        # TODO may need to do some sort of file check?  NIST expects png, but may be best to abort here if different format?
        return os.path.join(prefix,from_mask_name), os.path.join(prefix,to_mask_name)
    except:
        return "", ""

def get_status(data):
    try:
        status = data["imgManip"]["optOut"]
        return status_map[status]
    except:
        return "Processed"

def get_mdl(data):
    dict_list = []
    csv_dict = OrderedDict()
    mask_list=[]
    for row in data:
            # csv_dict["ProbeFileID"] = row['imgManipReq']["image"]["uri"].split("/")[-1].split(".")[0]
            csv_dict["ProbeFileID"] = getID(row['imgManipReq']["image"]["uri"])
            if 'imgManip' in row:
                csv_dict["ConfidenceScore"] = row['imgManip']["score"]
                csv_dict["OutputProbeMaskFilename"] = get_mdl_mask_name(row)
                csv_dict["ProbeStatus"] = get_status(row)
                mask_list.append({"path":row["imgManip"]["localization"]["mask"]["uri"],"name":get_mdl_mask_name(row)})
                csv_dict["ProbeOptOutPixelValue"] = ""

            else:
                csv_dict["ConfidenceScore"] = 0.0
                csv_dict["OutputProbeMaskFilename"] = ""
                csv_dict["ProbeStatus"] = "NonProcessed"
                csv_dict["ProbeOptOutPixelValue"] = ""

            dict_list.append(csv_dict)
    return dict_list, mask_list

def get_sdl(data):
    dict_list = []
    csv_dict = OrderedDict()
    mask_list=[]
    for row in data:
            csv_dict["ProbeFileID"] = getID(row['imgSpliceReq']["probeImage"]["uri"])
            csv_dict["DonorFileID"] = getID(row['imgSpliceReq']["donorImage"]["uri"])
            if 'imgSplice' in row:
                csv_dict["ConfidenceScore"] = row['imgSplice']["link"]["score"]
                from_mask, to_mask = get_sdl_mask_names(row)
                csv_dict["OutputProbeMaskFilename"] = to_mask
                csv_dict["OutputDonorMaskFilename"] = from_mask
                csv_dict["ProbeStatus"] = get_status(row)
                csv_dict["DonorStatus"] = get_status(row)
                # TODO Handle OptOut Mask
                mask_list.append({"path":row["imgSplice"]["link"]["toMask"]["mask"]["uri"],"name":to_mask})
                mask_list.append({"path":row["imgSplice"]["link"]["fromMask"]["mask"]["uri"],"name":from_mask})
                csv_dict["ProbeOptOutPixelValue"] = ""
                csv_dict["DonorOptOutPixelValue"] = ""

            else:
                csv_dict["ConfidenceScore"] = 0.0
                csv_dict["OutputProbeMaskFilename"] = ""
                csv_dict["OutputDonorMaskFilename"] = ""
                csv_dict["ProbeStatus"] = "NonProcessed"
                csv_dict["DonorStatus"] = "NonProcessed"
                csv_dict["ProbeOptOutPixelValue"] = ""
                csv_dict["DonorOptOutPixelValue"] = ""

            dict_list.append(csv_dict)
    return dict_list, mask_list

def get_vdl(data):
    # TODO
    "ProbeFileID|ConfidenceScore|ProbeStatus|VideoFrameSegments|AudioSampleSegments|VideoFrameOptOutSegments|OutputProbeMaskFilename|ProbeOptOutPixelValue"
    dict_list = []
    csv_dict = OrderedDict()
    mask_list=[]
    for row in data:
            csv_dict["ProbeFileID"] = getID(row['vidManipReq']["video"]["uri"])
            if 'vidManip' in row:
                csv_dict["ConfidenceScore"] = row['vidManip']["score"]
                csv_dict["ProbeStatus"] = get_status(row)
                f_det, a_det, f_opt = get_vdl_mask_names(row)
                csv_dict["VideoFrameSegments"] = f_det
                csv_dict["AudioSampleSegments"] = a_det
                csv_dict["VideoFrameOptOutSegments"] = f_opt
                #TODO Process VideoMask File
                csv_dict["OutputProbeMaskFilename"] = ""
                # mask_list.append({"path":row["imgManip"]["localization"]["mask"]["uri"],"name":get_mask_name(row)})
                csv_dict["ProbeOptOutPixelValue"] = ""

            else:
                csv_dict["ConfidenceScore"] = 0.0
                csv_dict["ProbeStatus"] = "NonProcessed"
                csv_dict["VideoFrameSegments"] = ""
                csv_dict["AudioSampleSegments"] = ""
                csv_dict["VideoFrameOptOutSegments"] = ""
                csv_dict["OutputProbeMaskFilename"] = ""
                csv_dict["OutputProbeMaskFilename"] = ""
                csv_dict["ProbeOptOutPixelValue"] = ""

            dict_list.append(csv_dict)
    return dict_list, mask_list

def parse_json(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = [json.loads(x) for x in lines]
    # pprint(data)


    if 'imgManip' in  data[0]:
        # Make MDL CSV
        return  get_mdl(data)
    elif 'imgSplice'  in data[0]:
        return get_sdl(data)
    elif 'vidManip' in data [0]:
        return get_vdl(data)
    elif 'imgMeta':
        #TODO
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    # TODO Other manip types\
    # return dict_list, mask_list

def write_nist_csv(data, csv_path):
    try:
        keys = data[0].keys()
        with open(csv_path, 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        return True
    except Exception as e:
        print(e)
        return False

def make_tarfile(out_filename, data, description = "", in_mem=True):
    if in_mem:
        tgz_name = out_filename + ".tgz"
        dir_name = out_filename
        with tarfile.open(tgz_name, "w:gz") as tar:
            # create top level directory whose name matches tar name
            top_dir = tarfile.TarInfo(dir_name)
            top_dir.type = tarfile.DIRTYPE
            top_dir.mode = 0o777
            tar.addfile(top_dir)
            # Create mask directory.  Might be empty
            mask_dir = tarfile.TarInfo(os.path.join(dir_name,"mask"))
            mask_dir.type = tarfile.DIRTYPE
            mask_dir.mode = 0o777
            tar.addfile(mask_dir)
            # add the csv file and json
            tar.add(data["csv"], arcname = os.path.join(dir_name,data["csv"]))
            json_path = data["json"].split("/")[-1]
            tar.add(data["json"], arcname = os.path.join(dir_name, json_path))
            # add masks
            for mask in data["masks"]:
                tar.add(mask["path"], arcname=os.path.join(dir_name,mask["name"]))
            # add text file
            with open (out_filename + ".txt", "w") as f:
                f.write(description)
            tar.add(out_filename+".txt", arcname=os.path.join(dir_name, out_filename+".txt"))



    else:
        raise NotImplementedError()

def remove_extension(filename):
    if filename.split(".")[-1] ==filename:
        return filename
    else:
        return filename.split(".")[0]

def main(args):
    # Get CSV and mask file info from JSON output
    outfile, ext = os.path.splitext(args.output)
    results, masks = parse_json(args.file)
    csv_path =  outfile+".csv"

    if write_nist_csv(results, csv_path):
        data = {
            "csv": csv_path,
            "masks": masks,
            "json": args.file
        }
        print(data)
        make_tarfile(outfile, data)
    else: print("Oops!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file",help="Path to the file containing the JSON output.")
    parser.add_argument("-o","--output", help="Name of the output tarfile, csv, etc.")

    args = parser.parse_args()
    main(args)

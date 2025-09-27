import ROOT
import array
import os


#input_file = "/lustre/work/jweijie/testbeam_round2/run1357_250924175021.root"
#output_file = "/lustre/work/jweijie/testbeam_round2/run1357_250924175021_converted.root"

def convertROOT(input_file, output_file):
    if output_file is None:
        output_file = input_file.replace(".root", "_converted.root")
        output_file = output_file.replace("jdamgov", "yofeng")
    print(f"Converting {input_file} to {output_file}")
    tree_name = "EventTree"

    # Open input file and tree
    f_in = ROOT.TFile.Open(input_file)
    tree_in = f_in.Get(tree_name)

    # Create output file and new tree
    f_out = ROOT.TFile(output_file, "RECREATE")
    tree_out = ROOT.TTree(tree_name, "Converted tree")

    # Storage for buffers and vectors
    buffers = {}
    vectors = {}

    # ROOT Type mapping: ROOT type → (array.array code, branch format, C++ type string)
    type_map = {
        "Float_t":     ('f', 'F', 'float'),
        "Double_t":    ('d', 'D', 'double'),
        "Int_t":       ('i', 'I', 'int'),
        "UInt_t":      ('I', 'i', 'unsigned int'),
        "Short_t":     ('h', 'S', 'short'),
        "UShort_t":    ('H', 's', 'unsigned short'),
        "Long64_t":    ('q', 'L', 'long long'),
        "ULong64_t":   ('Q', 'l', 'unsigned long long'),
        "ULong_t":     ('L', 'l', 'unsigned long'),
    }

    # Loop over all branches
    for br in tree_in.GetListOfBranches():
        br_name = br.GetName()
        cls_name = br.GetClassName()

        # Safely access the first leaf
        leaves = br.GetListOfLeaves()
        if not leaves or leaves.GetSize() == 0:
            print(f"Skipping {br_name}: no leaves found.")
            continue

        leaf = leaves.At(0)
        if not leaf:
            print(f"Skipping {br_name}: leaf is null.")
            continue

        leaf_title = leaf.GetTitle()
        leaf_type = leaf.GetTypeName()

        # Case 1: std::vector branches (already OK)
        if cls_name.startswith("vector"):
            print(f"Preserving vector branch: {br_name}")
            cpp_type = type_map.get(leaf_type, (None, None, None))[2]
            if cpp_type is None:
                print(f"Unsupported vector type {leaf_type} in {br_name}")
                continue
            vec = ROOT.std.vector(cpp_type)()
            vectors[br_name] = vec
            tree_out.Branch(br_name, vec)

        # Case 2: Leaf list (C-array)
        elif "[" in leaf_title:
            try:
                size = int(leaf_title.split('[')[1].split(']')[0])
            except Exception:
                print(f"Could not parse array size from leaf title: {leaf_title}")
                continue

            if leaf_type not in type_map:
                print(f"Unsupported type '{leaf_type}' for array {br_name}")
                continue

            arr_code, _, cpp_type = type_map[leaf_type]
            print(
                f"Converting C-array branch: {br_name} [{size}] as vector<{cpp_type}>")

            buf = array.array(arr_code, [0] * size)
            vec = ROOT.std.vector(cpp_type)()
            buffers[br_name] = buf
            vectors[br_name] = vec

            tree_in.SetBranchAddress(br_name, buf)
            tree_out.Branch(br_name, vec)

        # Case 3: Scalar primitive branch
        else:
            if leaf_type not in type_map:
                print(f"Unsupported scalar type '{leaf_type}' for {br_name}")
                continue

            arr_code, fmt_code, _ = type_map[leaf_type]
            print(f"Preserving scalar branch: {br_name} ({leaf_type})")
            buf = array.array(arr_code, [0])
            buffers[br_name] = buf

            tree_in.SetBranchAddress(br_name, buf)
            tree_out.Branch(br_name, buf, f"{br_name}/{fmt_code}")

    # Event loop: copy or convert all branches
    n_entries = tree_in.GetEntries()
    for i in range(n_entries):
        if i > 100000000:
            break
        if i % 1000 == 0:
            print(f"Processing entry {i}/{n_entries}")
        tree_in.GetEntry(i)
        for br_name in vectors:
            if br_name in buffers:
                vec = vectors[br_name]
                vec.clear()
                for v in buffers[br_name]:
                    vec.push_back(v)
        tree_out.Fill()

    # Write and close files
    tree_out.Write()
    f_out.Close()
    f_in.Close()

    print(f"Done. Converted tree saved to {output_file}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert ROOT data format.")
    #parser.add_argument("-i", "--input", type=str, default=input_file, help="Input ROOT file path")
    #parser.add_argument("-o", "--output", type=str, default=None, help="Output ROOT file path")
    #args = parser.parse_args()
    #input_file = "/lustre/work/jweijie/testbeam_round2/run1364_250924213612.root"
    #output_file = "/lustre/work/akshriva/Dream/testbeam02/run1364_250924213612_converted.root"
    #convertROOT(input_file, output_file)
    input_dir = "/lustre/work/jweijie/testbeam_round2"
    output_dir = "/lustre/work/akshriva/Dream/testbeam02"

    # Last 5 root files you listed
    files = [
     
        "run1365_250924220613.root"
    ]



    # Loop over files and call convertROOT
    for fname in files:
        input_file = os.path.join(input_dir, fname)
        # Add _converted.root to the output filename
        output_file = os.path.join(output_dir, fname.replace(".root", "_converted.root"))

        print(f"Converting: {input_file} -> {output_file}")
        convertROOT(input_file, output_file)

    

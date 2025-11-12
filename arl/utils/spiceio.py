import re

import pdb

prefix_dict = {
    'NMOS'  : 'm', 
    'PMOS'  : 'm', 
    'LDNMOS': 'm', 
    'LDPMOS': 'm',
    'PNP'   : 'q',
    'NPN'   : 'q',
    'DIODE' : 'd',
    'RES'   : 'r',
    'CAP'   : 'c',
    'IND'   : 'l'
}

predefined_cks = {
    'pmos_rvt': 'PMOS',
    'nmos_rvt': 'NMOS',
    'n11ll_ckt': 'NMOS',
    'p11ll_ckt': 'PMOS',
    'nhvt11ll_ckt': 'NMOS',
    'phvt11ll_ckt': 'PMOS',
    'nlvt11ll_ckt': 'NMOS',
    'plvt11ll_ckt': 'PMOS',
    'nt11ll_ckt': 'NMOS',
    'n25ll_ckt': 'NMOS',
    'nod33ll_ckt': 'NMOS',
    'nud18ll_ckt': 'NMOS',
    'p25ll_ckt': 'PMOS',
    'pod33ll_ckt': 'PMOS',
    'pud18ll_ckt': 'PMOS',
    'nt25ll_ckt': 'NMOS',
    'ntod33ll_ckt': 'NMOS',
    'ntud18ll_ckt': 'NMOS',
    'n11ll_dnw_ckt': 'NMOS',
    'nhvt11ll_dnw_ckt': 'NMOS',
    'nlvt11ll_dnw_ckt': 'NMOS',
    'n25ll_dnw_ckt': 'NMOS',
    'nod33ll_dnw_ckt': 'NMOS',
    'nud18ll_dnw_ckt': 'NMOS',
    'pnp11a100ll_ckt': 'PNP',
    'pnp11a25ll_ckt': 'PNP',
    'pnp11a4ll_ckt': 'PNP',
    'npn11a100ll_ckt': 'NPN',
    'npn11a25ll_ckt': 'NPN',
    'npn11a4ll_ckt': 'NPN',
    'pnp25a100ll_ckt': 'PNP',
    'pnp25a25ll_ckt': 'PNP',
    'pnp25a4ll_ckt': 'PNP',
    'npn25a100ll_ckt': 'NPN',
    'npn25a25ll_ckt': 'NPN',
    'npn25a4ll_ckt': 'NPN',
    'pnp11a100ll_sba_ckt': 'PNP',
    'pnp11a25ll_sba_ckt': 'PNP',
    'pnp11a4ll_sba_ckt': 'PNP',
    'npn11a100ll_sba_ckt': 'NPN',
    'npn11a25ll_sba_ckt': 'NPN',
    'npn11a4ll_sba_ckt': 'NPN',
    'pnp25a100ll_sba_ckt': 'PNP',
    'pnp25a25ll_sba_ckt': 'PNP',
    'pnp25a4ll_sba_ckt': 'PNP',
    'npn25a100ll_sba_ckt': 'NPN',
    'npn25a25ll_sba_ckt': 'NPN',
    'npn25a4ll_sba_ckt': 'NPN',
    'rpposab_2t_ckt' : 'RES',
    'mom_2t_ckt': 'CAP',
    'mom_3t_ckt': 'CAP',
}

def preprocess_predefined_cktfile(filename):
    circuit_types = {
        'ldn': 'LDNMOS',
        'ldp': 'LDPMOS',
        'pnp': 'PNP',
        'npn': 'NPN',
        'diode': 'DIODE',
        'res': 'RES',
        'cap': 'CAP',
        'ind': 'IND',
        'n': 'NMOS',
        'p': 'PMOS',
    }

    subckt_to_circuit_type = {}
    with open(filename) as f:
        subckt_names = f.readlines()

        for subckt in subckt_names:
            subckt = subckt.strip()

            for prefix, circuit_type in circuit_types.items():
                if subckt.startswith(prefix):
                    subckt_to_circuit_type[subckt] = circuit_type
                    break
            else:
                subckt_to_circuit_type[subckt] = 'Unknown'

        print(str(subckt_to_circuit_type))


def preprocess_spicefile(filename):
    # use re expression to accelerate the process
    pattern = r'|'.join(map(re.escape, predefined_cks.keys()))
    regex = re.compile(pattern)

    revised_lines = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            # machted with existing lines
            matched_re = regex.search(line)
            if matched_re:
                leading_spaces = len(line) - len(line.lstrip())
                line = line.lstrip()

                if line[0].lower() == 'x':
                    prefixed_line = ' ' * leading_spaces + prefix_dict[predefined_cks[matched_re.group()]] + line[1:]
                else:
                    prefixed_line = ' ' * leading_spaces + line

                revised_lines.append(prefixed_line)
            else:
                revised_lines.append(line)

    print('\n'.join(revised_lines))

    # Write the revised lines back to the file
    with open(filename, 'w') as f:
        f.writelines(revised_lines)

def extract_core_list(filename):
    # extract core devices and non core devices
    core_devices = []
    non_core_devices = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        flag_core = False
        flag_non_core = False

        for line in lines:
            core_index = line.find('** ## CORE:')
            non_core_index = line.find('** ## NON-CORE:')
    
            if  core_index != -1:
                for device in line[core_index + len('** ## CORE:'):].strip().split(','):
                    core_devices.append(device.strip().lower())
            elif non_core_index != -1:
                for device in line[non_core_index + len('** ## NON-CORE:'):].strip().split(','):
                    non_core_devices.append(device.strip().lower())

            if flag_core and flag_non_core:
                break

    return core_devices, non_core_devices


if __name__ == '__main__':
    # preprocess_predefined_cktfile('./circuit_names.txt')
    # preprocess_spicefile('/home/pxu/workspace/Istar/clustering/test_cases/netlist-ts2')
    print(extract_core_list('/home/pxu/workspace/Istar/clustering/test_cases/input1.ckt'))
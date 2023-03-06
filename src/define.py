MOLEXCLUDED = ['W', 'NA', 'CL']
RESEXCLUDED = ['W', 'ION']

LIPIDDEF = {
    # Phosphatidylcholine
    'PC':[
        'DTPC',
        'DLPC',
        'DPPC',
        'DBPC',
        'DXPC',
        'DYPC',
        'DVPC',
        'DOPC',
        'DIPC',
        'DFPC',
        'DGPC',
        'DAPC',
        'DRPC',
        'DNPC',
        'LPPC',
        'POPC',
        'PGPC',
        'PEPC',
        'PIPC',
        'PAPC',
        'PUPC',
        'PRPC',
        'POPX',
        'PIPX',
    ],

    # Phosphatidylethanolamine
    'PE':[
        'DTPE',
        'DLPE',
        'DPPE',
        'DBPE',
        'DXPE',
        'DYPE',
        'DVPE',
        'DOPE',
        'DIPE',
        'DFPE',
        'DGPE',
        'DAPE',
        'DUPE',
        'DRPE',
        'DNPE',
        'LPPE',
        'POPE',
        'PGPE',
        'PQPE',
        'PIPE',
        'PAPE',
        'PUPE',
        'PRPE',
    ],

    # Phosphatidylserine
    'PS':[
        'DTPS',
        'DLPS',
        'DPPS',
        'DBPS',
        'DXPS',
        'DYPS',
        'DVPS',
        'DOPS',
        'DIPS',
        'DFPS',
        'DGPS',
        'DAPS',
        'DUPS',
        'DRPS',
        'DNPS',
        'LPPS',
        'POPS',
        'PGPS',
        'PQPS',
        'PIPS',
        'PAPS',
        'PUPS',
        'PRPS',
    ],

    #Phosphatidylglycerol
    'PG':[
        'DTPG',
        'DLPG',
        'DPPG',
        'DBPG',
        'DXPG',
        'DYPG',
        'DVPG',
        'DOPG',
        'DIPG',
        'DFPG',
        'DGPG',
        'DAPG',
        'DRPG',
        'DNPG',
        'LPPG',
        'POPG',
        'PGPG',
        'PIPG',
        'PAPG',
        'PRPG',
        'OPPG',
        'JPPG',
        'JFPG',
    ],

    # Phosphatidic acid
    'PA':[
        'DTPA',
        'DLPA',
        'DPPA',
        'DBPA',
        'DXPA',
        'DYPA',
        'DVPA',
        'DOPA',
        'DIPA',
        'DFPA',
        'DGPA',
        'DAPA',
        'DRPA',
        'DNPA',
        'LPPA',
        'POPA',
        'PGPA',
        'PIPA',
        'PAPA',
        'PUPA',
        'PRPA',
    ],

    # phosphatidylinositol
    'PI':[
        'DPPI',
        'PVPI',
        'POPI',
        'PIPI',
        'PAPI',
        'PUPI',
        'DPP1',
        'PVP1',
        'POP1',
        'DPP2',
        'PVP2',
        'POP2',
        'PVP3',
        'POP3',
    ],

    # Glycerols
    'AG':[
        'PVDG',
        'PODG',
        'PIDG',
        'PADG',
        'PUDG',
        'TOG',
    ],

    # Lysophosphatidylcholine
    'LPC':[
        'CPC',
        'TPC',
        'LPC',
        'PPC',
        'VPC',
        'OPC',
        'IPC',
        'APC',
        'UPC',
    ],

    # Sphingomyelin
    'SM':[
        'DPSM',
        'DBSM',
        'DXSM',
        'PVSM',
        'POSM',
        'PGSM',
        'PNSM',
        'BNSM',
        'XNSM',
    ],

    # Ceramide
    'CER':[
        'DPCE',
        'DXCE',
        'PNCE',
        'XNCE',
    ],

    # Glycosphingolipids
    'GS':[
        'DPGS',
    ],

    # monosialotetrahexosylganglioside & monosialodihexosylganglioside 
    'GM':[
        'DPG1',
        'DBG1',
        'DXG1',
        'PNG1',
        'XNG1',
        'DPG3',
        'DBG3',
        'DXG3',
        'PNG3',
        'XNG3',
    ],

    # Glycoglycerolipids
    'DG':[
        'DPMG',
        'OPMG',
        'DFMG',
        'FPMG',
        'DPSG',
        'OPSG',
        'FPSG',
        'DPGG',
        'OPGG',
        'DFGG',
        'FPGG',
    ],

    # Sterols
    'sterols':[
        'CHOL',
        'CHOA',
        'CHYO',
        'ERGO',
        'HBHT',
        'HDPT',
        'HHOP',
        'HOPR',
    ],

    # Fatty acids
    'FA':[
        'LCA',
        'PCA',
        'BCA',
        'XCA',
        'ACA',
        'UCA',
    ]
}

NAME2TYPE = {}
for key in LIPIDDEF:
    for name in LIPIDDEF[key]:
        #lipidType.append(key)
        #lipidName.append(name)
        NAME2TYPE[name] = key
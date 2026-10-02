SOLVER_REGISTRY = {

    'ArterialNetwork': {
        'class':  [ 
            'lib.solvers.arterial_network',  
            'ArterialNetwork' 
        ],
        'models': [
            'Artery'
        ],
        'mpi':    [ 
            'lib.mpi.artery_mpi', 
            'ArteryMPI' 
        ]
    },
    
    'ArterialNetworkWK': {
        'class':  [ 
            'lib.solvers.arterial_network_wk',  
            'ArterialNetworkWK' 
        ],
        'solvers': [
            'ArterialWindkessel'
        ],
        'models': [
            'Windkessel'
        ],
    },
    
    'VariableArterialWK': {
        'class':  [ 
            'lib.solvers.variable_arterial_wk',  
            'VariableArterialWK' 
        ],
        'solvers': [
            'ArterialWindkessel'
        ],
        'models': [
            'VariableWindkessel'
        ],
    },

    'SkeletalMuscle': {
        'class':  [ 
            'lib.solvers.skeletal_muscle',  
            'SkeletalMuscle' 
        ],
        'models': [
            'OxygenTransport',
            'CellularMetabolism'
        ]
    },

    'DirectWindkesselTone': {
        'class':  [ 
            'lib.solvers.direct_windkessel_tone',  
            'DirectWindkesselTone' 
        ],
        'models': [
            'Windkessel',
            'DirectToneRegulation'
        ]
    },

    'DirectNetworkTone': {
        'class':  [ 
            'lib.solvers.direct_network_tone',  
            'DirectNetworkTone' 
        ],
        'models': [
            'MicrovascularNetwork',
            'DirectToneRegulation'
        ]
    },

    'NetworkTone': {
        'class':  [ 
            'lib.solvers.network_tone',  
            'NetworkTone' 
        ],
        'models': [
            'MicrovascularNetwork',
            'ToneRegulation'
        ]
    },

    'DirectMuscleTone': {
        'class':  [ 
            'lib.solvers.direct_muscle_tone',  
            'DirectMuscleTone' 
        ],
        'solvers': [
            'DirectNetworkAutoregulation',
            'SkeletalMuscle'
        ],
    },

    'MuscleTone': {
        'class':  [ 
            'lib.solvers.muscle_tone',  
            'MuscleTone' 
        ],
        'solvers': [
            'NetworkAutoregulation',
            'SkeletalMuscle'
        ],
    },

    'DirectMicrovascularTone': {
        'class':  [ 
            'lib.solvers.direct_microvascular_tone',  
            'DirectMicrovascularTone' 
        ],
        'models': [
            'Windkessel',
            'MicrovascularNetwork',
            'DirectToneRegulation'
        ]
    },

    'MicrovascularTone': {
        'class':  [ 
            'lib.solvers.microvascular_tone',  
            'MicrovascularTone' 
        ],
        'models': [
            'Windkessel',
            'MicrovascularNetwork',
            'ToneRegulation'
        ]
    },

    'WindkesselTone': {
        'class':  [ 
            'lib.solvers.windkessel_tone',  
            'WindkesselTone' 
        ],
        'models': [
            'VariableWindkessel',
            'VectorisedToneRegulation'
        ]
    },

    'PADOVA': {
        'class':  [ 
            'lib.solvers.padova',  
            'PADOVA' 
        ],
        'solvers': [
            'ArterialNetworkWK',
            'MuscleTone'
        ],
    },

}

MODEL_REGISTRY = {
    'Artery': {
        'class':  ( 
            'lib.models.artery',  
            'Artery'
        ),
        'io':     (
            'lib.io.artery_io', 
            'ArteryIO' 
        ),
    },

    'Windkessel': {
        'class':  ( 
            'lib.models.windkessel',  
            'Windkessel'
        ),
        'io':     (
            'lib.io.windkessel_io', 
            'WindkesselIO' 
        ),
    },

    'MicrovascularNetwork': {
        'class':  ( 
            'lib.models.artery',  
            'MicrovascularNetwork'
        ),
        'io':     (
            'lib.io.network_io', 
            'NetworkIO' 
        ),
    },

    'RelativeToneRegulation': {
        'class':  ( 
            'lib.models.relative_tone_regulation',  
            'RelativeToneRegulation'
        ),
        'io':     (
            'lib.io.tone_io', 
            'ToneIO' 
        ),
    },

    'DirectToneRegulation': {
        'class':  ( 
            'lib.models.direct_tone_regulation',  
            'DirectToneRegulation'
        ),
        'io':     (
            'lib.io.tone_io', 
            'ToneIO' 
        ),
    },

    'ToneRegulation': {
        'class':  ( 
            'lib.models.tone_regulation',  
            'ToneRegulation'
        ),
        'io':     (
            'lib.io.tone_io', 
            'ToneIO' 
        ),
    },

    'OxygenTransport': {
        'class':  ( 
            'lib.models.oxyge_transport',  
            'OxygenTransport'
        ),
        'io':     (
            'lib.io.oxygen_io', 
            'OxygenIO' 
        ),
    },

    'CellularMetabolism': {
        'class':  ( 
            'lib.models.cellular_metabolism',  
            'CellularMetabolism'
        ),
        'io':     (
            'lib.io.metabolism_io', 
            'MetabolismIO' 
        ),
    },

}
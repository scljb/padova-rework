"""
auth: L. J. Barratt
Created on 14/09/26

Class housing methods for input and output processing.
"""

#%% PACKAGES

#* EXTERNAL PACKAGES
from fenics        import project
from numpy         import column_stack, savetxt 
from pandas        import read_excel

#* INTERNAL PACKAGES
from lib.io.new.base_io import BaseIO


#%% CLASS

class ArteryIO(BaseIO):
        
    inputs  = ['solution', 'fluid', 'network']

    def __init__(self, file):

        super().__init__(file)

        # # READ FROM CSV FILE FOR NETWORK DATA
        self.read_from_csv()

        # ? Check length of parameter lists against given number of arteries
        expected_len = self.network['number_of_arteries'][0]
        for key, value in self.vessel.items():
            if key != 'nu':
                if len(value) != expected_len:
                    raise ValueError("vessel['{key}'] has wrong length")


    def read_from_csv(self):

        # # CHECK FOR FILE NAME   
        if 'source' not in self.network:
            raise ValueError('.xlsx file not found in case directory')

        # # LOAD FILE
        file_path = f'{self.file.parent}/{self.network['source'][0]}.xlsx'
        df = read_excel(file_path, skiprows=1, engine='openpyxl')

        # # STUFF
        # * Select columns for parameter selection
        parameter_columns = [col for i, col in enumerate(df.columns) if i >= 3]
        variable_names = [
            ( (col.split(', ')[1].split(' [')[0])
            if ', ' in col else (col.split(' [')[0]) ).lower()
            for col in parameter_columns
        ]

        # * Create vessel dict
        vessel = {}
        for var, col in zip(variable_names[:12], parameter_columns[:12]):
            if var != 'state':
                vessel[var] = df[col].astype(float).tolist()

            else:
                vessel[var] = [
                    str(x).strip().lower() 
                    for val in df[col].tolist() 
                    for x in str(val).split(',')
                ]


        # if the possion's ratio is not given, assume incompressible
        if 'nu' not in vessel:
            vessel['nu'] = [0.5]


        # * Create outflow dict
        filtered_df = df[df['Type'].str.strip().str.lower() != 'junction']

        outflow = {}
        for var, col in zip(variable_names[12:], parameter_columns[12:]):
            if var != 'type':
                outflow[var] = filtered_df[col].astype(float).tolist()
            else:
                outflow[var] = [
                    str(x).strip().lower()
                    for val in filtered_df[col].tolist()
                    for x in str(val).split(',')
                ]

        # * Identify Roots/Parents/Daughters/Terminals
        id_columns = [col for i, col in enumerate(df.columns) if i < 3]
        ids = df[id_columns[0]].tolist()
        pid = df[id_columns[-1]].tolist()

        roots     = []  # for inlets
        parents   = {}  # for junctions
        terminals = []  # for outlets
        for child, parent in zip(ids, pid):

            # child is root
            if child == parent:
                roots.append(parent)

            # child is non-root 
            else: # parents/daughers
                parents.setdefault(parent, []).append(child)

            # child is terminal
            if df.loc[child, 'Type'].lower() != 'junction':
                terminals.append(child)

        # # FIN
        self.vessel  = vessel
        self.outflow = outflow
        self.network['roots']     = roots
        self.network['parents']   = parents
        self.network['terminals'] = terminals


    def write_solution(self, step, time, solver, **stuff):
        '''
        Intended to be used via the ArterialNetwork solver class.
        '''

        for artery in solver.local_arteries:
            self.write_arterial_solution(artery, time)


    def write_arterial_solution(self, artery, time):

        # # PROCESS DATA
        area, vel = artery.Un.split()
        artery.update_pressure()
        pres = project(artery.pn,artery.W)

        x    = artery.mesh.coordinates()
        aver = area.compute_vertex_values(artery.mesh)
        vver = vel.compute_vertex_values(artery.mesh)
        pver = pres.compute_vertex_values(artery.mesh)
        
        data = column_stack((x, aver, vver, pver))

        # # PRINT TO FILE
        idx = artery.artery_id
        filename = self.output_dir / f'artery_{idx}/t{time:.6f}.txt'
        savetxt(filename, data, header="x A u p", comments='')
        


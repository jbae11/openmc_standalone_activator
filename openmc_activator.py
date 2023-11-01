# activate it using ORIGEN
import os
import numpy as np
import openmc
import openmc.deplete
import h5py
from openmc.data import atomic_mass, AVOGADRO
import json

def get_openmc_reactions():
    from openmc.deplete.chain import REACTIONS
    d = {k:min(v.mts) for k,v in REACTIONS.items()}
    return d

class OpenMCActivator:
    def __init__(self, ebins: list, mg_flux: list):
        assert(len(ebins) == len(mg_flux) + 1)
        
        self.ebins = np.array(ebins)
        # get midpoints
        self.ebins_midpoints = (self.ebins[:-1] + self.ebins[1:])/2
        self.mg_flux = np.array(mg_flux)
        # normalize
        self.norm_mg_flux = self.mg_flux / sum(self.mg_flux)
        
    def activate(self, mass_dict, flux_mag_list,
                 timestep_list,
                chain_file_path,
                openmc_neutron_data_dir='/home/4ib/git/openmc/data/endfb71_hdf5',
                # fuel_volume,
                u_timestep='d',
                metric_list=['mass']
                ):
        openmc.config['chain_file'] = chain_file_path
        # 0. setup
        reactions = get_openmc_reactions()
        reactions['fission'] = 18
        mt_list= list(reactions.values())
        mt_names = list(reactions.keys())

        # 1. build flux-normalized microxs array
        data_files = os.listdir(openmc_neutron_data_dir)
        data_files = [q for q in data_files if not q.startswith('c_') and '.h5' in q]
        isos = [q.replace('.h5','') for q in data_files]
        # kill ones with long suspicious names
        isos = [q for q in isos if len(''.join([w for w in q if w.isalpha()])) > 2]
        
        microxs_arr = np.zeros((len(isos),len(mt_list)))

        for nuclideE, iso in enumerate(isos):
            nuclide = openmc.data.IncidentNeutron.from_hdf5(os.path.join(openmc_neutron_data_dir, iso+'.h5'))
            for rxn_type, mt in enumerate(mt_list):
                try:
                    total_xs = np.array(nuclide[mt].xs['294K'](self.ebins_midpoints))
                except:
                    total_xs = np.zeros(len(self.ebins_midpoints))

                total_norm = sum(total_xs*self.norm_mg_flux)
                microxs_arr[nuclideE,rxn_type] = total_norm


        # 2. Use scaling factor and timesteps 
        chain = openmc.deplete.Chain.from_xml(chain_file_path)
        # 3. Construct openmc MicroXS object from array
        # probably a smarter way to do this
        microxs = openmc.deplete.MicroXS.from_array(isos,mt_names,microxs_arr)


        # 3.5 Construct openmc material
        material = openmc.Material(material_id=1, name='dummy')
        # case sensitive
        mass_dict = {k.lower().capitalize():v for k,v in mass_dict.items()}
        tot_mass = sum(mass_dict.values())
        for k,v in mass_dict.items():
            material.add_nuclide(k, v/tot_mass, percent_type='wo')
        # material.add_components(mass_dict, 'wo')
        material.set_density('g/cc', tot_mass)
        material.depletable = True
        material.volume = 1.0

        # 4. Define the operator to perform depletion calculation
        materials = openmc.Materials([material])        
        operator = openmc.deplete.IndependentOperator(materials,microxs,chain_file_path)

        # 5. Use the operator to perform depletion
        predictor = openmc.deplete.PredictorIntegrator(operator, timestep_list,
                                                       source_rates=flux_mag_list,
                                                       timestep_units=u_timestep,
                                                       solver='cram48')
        # now integrate
        predictor.integrate()
        #! change this to in-memory
        results = openmc.deplete.Results(os.path.join(os.getcwd(), 'depletion_results.h5'))
        
        isos = list(h5py.File('depletion_results.h5')['nuclides'].keys())
        metric_dict = {}
        unit_dict = {'mass': 'grams', 'activity':'becquerels', 'decay_heat': 'watts',
                     'atom': ''}
        for metric in metric_list:
            key = metric + '_' + unit_dict[metric]
            metric_dict[key] = {}
            if metric == 'atom':
                for iso in isos:
                    time, vals = results.get_atoms(mat='1', nuc=iso)
                    metric_dict[key][iso] = vals
            
            # this 14.0 version
            # wish.com version
            elif metric == 'mass':
                for iso in isos:
                    time, atoms = results.get_atoms(mat='1', nuc=iso)
                    mass = atoms * atomic_mass(iso) / AVOGADRO
                    metric_dict[key][iso] = mass
        
            elif metric == 'activity':
                for indx, result in enumerate(results):
                    mat = result.get_material('1')
                    act_dict = mat.get_activity(units='Bq', by_nuclide=True)
                    if indx == 0:
                        metric_dict[key] = {k:[v] for k,v in act_dict.items()}
                    else:
                        for k,v in act_dict.items():
                            metric_dict[key][k].append(v)
            elif metric == 'decay_heat':
                for indx, result in enumerate(results):
                    mat = result.get_material('1')
                    act_dict = mat.get_decay_heat(units='W', by_nuclide=True)
                    if indx == 0:
                        metric_dict[key] = {k:[v] for k,v in act_dict.items()}
                    else:
                        for k,v in act_dict.items():
                            metric_dict[key][k].append(v)

        return metric_dict


if __name__ == '__main__':
    import pickle
    data = pickle.load(open('./irradiation_setup.pkl', 'rb'))
    locals().update(data)


    mass_dict = {'Ag107': 0.518, 'Ag109': 0.482}
    obj = OpenMCActivator(ebins, fluxes)
    #obj.activate(mass_dict, flux_mag_list[0], [days_list[0]],
    time, metric_dict = obj.activate(mass_dict, flux_mag_list[0], [days_list[0]], 
                        chain_file_path='/home/4ib/git/fermi/fermi/data/openmc/chain_endf_b8.0.xml',
                        # chain_file_path='/home/4ib/Downloads/chain_endfb71_pwr.xml',
                        metric_list=['atom', 'mass', 'activity', 'decay_heat'])
                        # metric_list=['mass', 'activity', 'decay_heat'])
    print(metric_dict)
    print(time)
import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo
import xfields as xf
# to make output folder in this script
import os
# to use parabolic longitudinal profile
from parabolic_longitudinal_distribution import parabolic_longitudinal_distribution
# to estimate the emittances from the distribution
from statisticalEmittance import *

####################
# Choose a context #
####################

context = xo.ContextCpu()
#context = xo.ContextCupy()
# context = xo.ContextPyopencl('0.0')

print(context)

#os.mkdir('output')

mad = Madx()
mad.call('psb_flat_bottom.madx')

line= xt.Line.from_madx_sequence(mad.sequence['psb1'])
line.particle_ref=xp.Particles(mass0=xp.PROTON_MASS_EV,
                               gamma0=mad.sequence.psb1.beam.gamma)

nemitt_x=1.5e-6
nemitt_y=1e-6
bunch_intensity=50e10
sigma_z=16.9

# from space charge example
num_turns=1 
num_spacecharge_interactions = 160 
tol_spacecharge_position = 1e-2 
n_part=int(100e3)

# Available modes: frozen/quasi-frozen/pic
mode = 'pic'

#############################################
# Install spacecharge interactions (frozen) #
#############################################

lprofile = xf.LongitudinalProfileQGaussian(
        number_of_particles=bunch_intensity,
        sigma_z=sigma_z,
        z0=0.,
        q_parameter=1.)

xf.install_spacecharge_frozen(line=line,
                   particle_ref=line.particle_ref,
                   longitudinal_profile=lprofile,
                   nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                   sigma_z=sigma_z,
                   num_spacecharge_interactions=num_spacecharge_interactions,
                   tol_spacecharge_position=tol_spacecharge_position)

#################################
# Switch to PIC or quasi-frozen #
#################################

if mode == 'frozen':
    pass # Already configured in line
elif mode == 'quasi-frozen':
    xf.replace_spacecharge_with_quasi_frozen(
                                    line,
                                    update_mean_x_on_track=True,
                                    update_mean_y_on_track=True)
elif mode == 'pic':
    pic_collection, all_pics = xf.replace_spacecharge_with_PIC(
        _context=context, line=line,
        n_sigmas_range_pic_x=8,
        n_sigmas_range_pic_y=8,
        nx_grid=128, ny_grid=128, nz_grid=64,
        n_lims_x=7, n_lims_y=3,
        z_range=(-3*sigma_z, 3*sigma_z))
else:
    raise ValueError(f'Invalid mode: {mode}')


#################
# Build Tracker #
#################

line.build_tracker(_context=context)
line_sc_off = line.filter_elements(exclude_types_starting_with='SpaceCh')


######################
# Generate particles #
######################

particles = parabolic_longitudinal_distribution(_context=context, num_particles=n_part,
                            total_intensity_particles=bunch_intensity,
                            nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z,
                            particle_ref=line.particle_ref,
                            line=line_sc_off)

# to estimate emittance if needed
r=StatisticalEmittance(context='CPU')
bunch_moments=r.measure_bunch_moments(particles)
print(bunch_moments['nemitt_x'])
print(bunch_moments['nemitt_y'])
output=[]

# to do your tracking & monitoring depending on needs 

# tracking and monitoring  & dumping all turns (not ideal for long term simulations but used for FMA for example)

#line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True )
#np.save('x',line.record_last_track.x)
#np.save('px',line.record_last_track.px)
#np.save('y',line.record_last_track.y)
#np.save('py',line.record_last_track.py)
#np.save('z',line.record_last_track.zeta)
#np.save('dp',line.record_last_track.delta)

# tracking, monitoring & saving distributions at some turns (useful for long term tracking)
# example below estimates emittances at each turn (not yet in the monitors needs the loop)


for i in range(num_turns):
    line.track(particles)
    bunch_moments=r.measure_bunch_moments(particles)
    output.append([len(r.coordinate_matrix[0]),bunch_moments['nemitt_x'].tolist(),bunch_moments['nemitt_y'].tolist(),bunch_moments['emitt_z'].tolist()])
    if i in range(-1, num_turns, 10000):
        # distribution saved through the emittance module, any other xsuite example could be used to dump the distribution
        if r.context=='GPU':
            np.save('output/distribution_'+str(int(i)), r.coordinate_matrix.get())
        else:
            np.save('output/distribution_'+str(int(i)), r.coordinate_matrix)

# to save all emittances 
ouput=np.array(output)
np.save('output/emittances', output)

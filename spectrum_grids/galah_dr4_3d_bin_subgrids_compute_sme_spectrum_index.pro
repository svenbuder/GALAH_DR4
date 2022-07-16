PRO galah_dr4_3d_bin_subgrids_compute_sme_spectrum_index,grid_index,spectrum_index,ccd

; For virtual machine mode
;                                                                                                                                            
cla=command_line_args(count=count)
if count ne 0 then begin
   grid_index = cla[0]
   spectrum_index  = cla[1]
   ccd = cla[2]
   endif
; Ensure strings without spaces
;                                                                                                                                            
grid_index     = strcompress(grid_index,/remove)
grid_index = fix(grid_index,type=int)
spectrum_index     = strcompress(spectrum_index,/remove)
spectrum_index = fix(spectrum_index,type=int)
ccd = strcompress(ccd,/remove)
ccd = fix(ccd,type=int)

print,grid_index,spectrum_index,ccd

all_grids = mrdfits('galah_dr4_model_trainingset_gridpoints.fits',1)

this_grid = where(all_grids.index_subgrid eq grid_index,nr_matching_grid_indices)

if nr_matching_grid_indices eq 1 then begin
   print,'Found matching grid index'
   
   grid_teff = all_grids[this_grid].teff_subgrid
   teff_string = string(grid_teff,format='(i4)')
   grid_logg = all_grids[this_grid].logg_subgrid
   if grid_logg ge 0.0 then logg_string = string(grid_logg,format='(f4.2)')
   if grid_logg lt 0.0 then logg_string = string(grid_logg,format='(f5.2)')
   grid_fe_h = all_grids[this_grid].fe_h_subgrid
   if grid_fe_h ge 0.0 then fe_h_string = string(grid_fe_h,format='(f4.2)')
   if grid_fe_h lt 0.0 then fe_h_string = string(grid_fe_h,format='(f5.2)')
   grid_name = teff_string+'_'+logg_string+'_'+fe_h_string

   grid_params = mrdfits('3d_bin_subgrids/'+grid_name+'/galah_dr4_trainingset_'+grid_name+'.fits',1)

endif else begin
   goto,end_of_script
endelse

log_file = '3d_bin_subgrids/'+grid_name+'/galah_dr4_trainingset_'+grid_name+'_'+fs(spectrum_index)+'_'+fs(ccd)+'.log'
sme_file = '3d_bin_subgrids/'+grid_name+'/galah_dr4_trainingset_'+grid_name+'_'+fs(spectrum_index)+'_'+fs(ccd)+'.out'

journal, log_file

restore,'3d_bin_subgrids_211102_template_ccd'+fs(ccd)+'.inp'

solar_abund = sme.abund
a_solar = sme_abundances(sme)

sme.teff = grid_params[spectrum_index].teff
sme.grav = grid_params[spectrum_index].logg
sme.feh  = grid_params[spectrum_index].fe_h
sme.vmic = grid_params[spectrum_index].vmic

if sme.grav ge 2.9 then sme.nlte[0].nlte_grids[2] = ['nlte_Li_dwarf_scatt_idlsme.grd']
if sme.grav lt 2.9 then sme.nlte[0].nlte_grids[2] = ['nlte_Li_giant_scatt_idlsme.grd']

abund = grid_params[spectrum_index].sme_abund
sme.abund = float(abund)

a_adjusted = sme_abundances(sme)

print,'Running on grid with index '+fs(grid_index)
print,'Running spectrum index '+fs(spectrum_index)+' with TEFF '+fs(sme.teff)+' LOGG '+fs(sme.grav)+' FEH '+fs(sme.feh)+' VMIC '+fs(sme.vmic)
print,'Adjusted [Fe/H] and sme.abund[25], A(X=25)'
print,sme.feh,sme.abund[25],a_adjusted[25]

elem = ['Li','C','N','O','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Rb','Sr','Y','Zr','Mo','Ru','Ba','La','Ce','Nd','Sm','Eu']
elem_i = [3,6,7,8,11,12,13,14,19,20,21,22,23,24,25,26,27,28,29,30,37,38,39,40,42,44,56,57,58,60,62,63]-1
print,'X, Elem, sme.abund,    A(X),        [X/H],       [X/Fe],      [X/Fe] (aim)'
for i=0,n_elements(elem)-1 do begin
   label = elem[i]+'_fe'
   print,fs(elem_i[i]+1),' ',elem[i],sme.abund[elem_i[i]],a_adjusted[elem_i[i]],a_adjusted[elem_i[i]]-a_solar[elem_i[i]],(a_adjusted[elem_i[i]]-a_solar[elem_i[i]]) - (a_adjusted[25]-a_solar[25])
endfor

; STEP 1: RUN LINE DEPTH TEST A LA NORDLANDER
; SME will check all DEPTHs no matter what wave we plug in
nw = 10 & wave = 0.076*findgen(nw) + median(sme.wave) & sob = dblarr(nw)+1 & uob = sob/100 & wran = minmax(wave) & mob = intarr(nw)+2 & mob[nw/3:2*nw/3] = 1 & wind = nw-1
; x_seg[0],x_seg[-1],vfact_seg[0],vfact_seg[-1],wave_seg[0],wave_set[-1]
; CCD1: 4810.0392       4810.0572       4820.0392       4820.0221
; smod_seg = resamp(x_seg*vfact_seg, y_seg, wave_seg)
linetest = modify_struct(sme, {wave: wave, sob:sob, uob:uob, wran:wran, mob:mob, wind:wind})
sme_main, linetest

; print Number of significant lines and make sure to include H1 and all atomic lines'
i = where(linetest.depth gt 0.001,ic)
print,'Number of significant lines',ic

i = where(linetest.depth gt 0.001 or linetest.species eq 'H 1',ic)
print,'Number of significant lines + H1',ic

i = where(linetest.depth gt 0.001 or linetest.species eq 'H 1' or ~is_molecule(linetest.species),ic)
print,'Number of significant lines + H1 + all atomic lines',ic

sme = modify_struct(sme, { $
      lineindices:i, $
      atomic:sme.atomic[*,i], $
      species:sme.species[i], $
      lande:sme.lande[i], $
      depth:linetest.depth[i], $
      origdepth:linetest.depth, $
      lineref:sme.lineref[i], $
      line_extra:sme.line_extra[*,i], $
      line_term_low:sme.line_term_low[i], $
      line_term_upp:sme.line_term_upp[i]} $
)

sme_main, sme, equidistant=300e3
;results = modify_struct(sme, deltags=['atomic','species','lande','lineref','line_extra','line_term_low','line_term_upp', 'sob', 'uob', 'mob'])
;save, results, file=sme_output

results = {wave: sme.wave, smod: sme.smod, cmod: sme.cmod, wint: sme.wint, sint: sme.sint}
save, results, file=sme_file

journal

end_of_script:

END

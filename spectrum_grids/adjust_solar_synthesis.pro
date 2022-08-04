PRO adjust_solar_synthesis,ccd

cla=command_line_args(count=count)
if count ne 0 then begin
   ccd = cla[0]
endif

log_file = 'sun_synthesis/galah_dr4_solar_synthesis_zeropoints_'+fs(ccd)+'.log'
sme_file = 'sun_synthesis/galah_dr4_solar_synthesis_zeropoints_'+fs(ccd)+'.out'

journal, log_file

restore,'3d_bin_subgrids_211102_template_ccd'+fs(ccd)+'.inp'

solar_abund = sme.abund
a_solar = sme_abundances(sme)

marcs_abund,marcs_abund_sme,eonh12=marcs_abund_onh12_original
marcs_abund,marcs_abund_sme,eonh12=marcs_abund_onh12

zeropoints = mrdfits('../spectrum_post_processing/galah_dr4_zeropoints.fits',1)

sme.teff = 5772.
sme.grav = 4.438
sme.feh  = 0.00
sme.vmic = 1.06

marcs_abund_onh12[3-1] = zeropoints.A_Li
marcs_abund_onh12[6-1] = zeropoints.A_C
marcs_abund_onh12[7-1] = zeropoints.A_N
marcs_abund_onh12[8-1] = zeropoints.A_O
marcs_abund_onh12[11-1] = zeropoints.A_Na
marcs_abund_onh12[12-1] = zeropoints.A_Mg
marcs_abund_onh12[13-1] = zeropoints.A_Al
marcs_abund_onh12[14-1] = zeropoints.A_Si
marcs_abund_onh12[19-1] = zeropoints.A_K
marcs_abund_onh12[20-1] = zeropoints.A_Ca
marcs_abund_onh12[21-1] = zeropoints.A_Sc
marcs_abund_onh12[22-1] = zeropoints.A_Ti
marcs_abund_onh12[23-1] = zeropoints.A_V
marcs_abund_onh12[24-1] = zeropoints.A_Cr
marcs_abund_onh12[25-1] = zeropoints.A_Mn
marcs_abund_onh12[26-1] = zeropoints.A_Fe
marcs_abund_onh12[27-1] = zeropoints.A_Co
marcs_abund_onh12[28-1] = zeropoints.A_Ni
marcs_abund_onh12[29-1] = zeropoints.A_Cu
marcs_abund_onh12[30-1] = zeropoints.A_Zn
marcs_abund_onh12[37-1] = zeropoints.A_Rb
marcs_abund_onh12[38-1] = zeropoints.A_Sr
marcs_abund_onh12[39-1] = zeropoints.A_Y
marcs_abund_onh12[40-1] = zeropoints.A_Zr
marcs_abund_onh12[42-1] = zeropoints.A_Mo
marcs_abund_onh12[44-1] = zeropoints.A_Ru
marcs_abund_onh12[56-1] = zeropoints.A_Ba
marcs_abund_onh12[57-1] = zeropoints.A_La
marcs_abund_onh12[58-1] = zeropoints.A_Ce
marcs_abund_onh12[60-1] = zeropoints.A_Nd
marcs_abund_onh12[62-1] = zeropoints.A_Sm
marcs_abund_onh12[63-1] = zeropoints.A_Eu

;Convert to abundances relative to total number of nuclei.
log_eonh = marcs_abund_onh12 - 12.0
eonh = 10d0 ^ log_eonh
renorm = total(eonh)
eontot = eonh / renorm

;SME expects Hydrogen abundance to NOT be logarithmic.
log_eontot = float(alog10(eontot))
abund = log_eontot
abund(0) = eonh(0) / renorm
abund = float(abund)

sme.abund = abund

a_adjusted = sme_abundances(sme)

print,'Running solar synthesis with TEFF '+fs(sme.teff)+' LOGG '+fs(sme.grav)+' FEH '+fs(sme.feh)+' VMIC '+fs(sme.vmic)
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

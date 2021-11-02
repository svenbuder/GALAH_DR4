PRO galah_dr4_3d_bin_subgrids_compute_sme_template,ccd=ccd

if not keyword_set(ccd) then ccd = 3

;if not_keyword_set(param_grid) then begin
teff = 5777.
logg = 4.44
feh = 0.0
vmic = 1.0
marcs_abund,abund,fehmod=feh

; BOOKKEEPING
outfile = '3d_bin_subgrids_211102_template_ccd'+fs(ccd)

; WAVELENGTH GRID with R = 300,000
; galah_master_v5.2.fits:
; 4675.0020 - 4949.9739
; 5625.0060 - 5899.9990
; 6425.0020 - 6774.9960
; 7550.0240 - 7924.9830
nseg = 1
R = 300e3
if ccd eq 1 then wave = [4676.1,4948.9] ; dispersion: 0.046 ; at R0.3M: 0.016
if ccd eq 2 then wave = [5625.1,5899.9] ; dispersion: 0.055 ; at R0.3M: 0.019
if ccd eq 3 then wave = [6425.1,6774.9] ; dispersion: 0.064 ; at R0.3M: 0.022
if ccd eq 4 then wave = [7550.1,7924.9] ; dispersion: 0.074 ; at R0.3M: 0.030
w1 = min(wave) - 1 & wn = max(wave) + 1
N = ceil((alog10(wn) - alog10(w1)) / (alog10(w1*(R+1)/R) - alog10(w1)))
wave = logrange(w1, wn, n)
sob = wave*0+1
uob = sob / 100
mob = intarr(n_elements(wave))
wran = [wave[0],wave[-1]]
wind  =  [long(n_elements(wave)-1)]

; LINELIST: stars from all of GALAH, and then select only those within *wave*
line_merge, 'galah_master_v5.2.fits', line_atomic, line_species, line_lande, line_depth, line_ref,term_low=line_term_low,term_upp=line_term_upp,short_format=short_format,extra=line_extra,lulande=line_lulande
j=[0]
for i=0,nseg-1 do begin
   k=where(line_atomic[2,*] ge wran[0,i] and line_atomic[2,*] le wran[1,i],kc)
   if kc ne 0 then j=[j,k] 
endfor
j=unique(j[1:*])
species   =  line_species  [j]
atomic    =  line_atomic [*,j]
lande     =  line_lande    [j]
depth     =  line_depth    [j]
depth[*] = 0.99 ; set all depth to 0.99 for depth test
lineref   =  line_ref      [j]
term_low  =  line_term_low [j]
term_upp  =  line_term_upp [j]
extra     =  line_extra  [*,j]
lulande   =  line_lulande[*,j]
gf_free=lonarr(n_elements(species))

; NLTE
nlte_elem_flags = bytarr(99)
nlte_grids = strarr(99)
nltee = ['H','Li','C','N','O','Na','Mg','Al','Si','K','Ca','Mn','Ba']
nltez = [ 1 ,  3 , 6 , 7 , 8 , 11 , 12,  13 , 14 , 19, 20 , 25 , 56 ]
for i=0,n_elements(nltez)-1 do begin
   inlte = where(nltee eq nltee[i]) & inlte=inlte[0]
   nlte_elem_flags[nltez[inlte]-1]  = 1B
   ;nlte_grids[nltez[inlte]-1]       = ['Amarsi19_']+nltee[inlte]+['.grd']
   ;if nltee[inlte] eq 'Li' and teff lt 5750. then nlte_grids[nltez[inlte]-1] = ['Amarsi19_']+nltee[inlte]+['_t3800_5750.grd']
   ;if nltee[inlte] eq 'Li' and teff ge 5750. then nlte_grids[nltez[inlte]-1] = ['Amarsi19_']+nltee[inlte]+['_t5750_8000.grd']
   nlte_grids[nltez[inlte]-1]       = ['nlte_']+nltee[inlte]+['_scatt_idlsme.grd']
   if nltee[inlte] eq 'C' then nlte_grids[nltez[inlte]-1] = ['nlte_C_ama51_idlsme.grd']
   if nltee[inlte] eq 'Li' and logg ge 3.0 then nlte_grids[nltez[inlte]-1] = ['nlte_Li_dwarf_scatt_idlsme.grd']
   if nltee[inlte] eq 'Li' and logg lt 3.0 then nlte_grids[nltez[inlte]-1] = ['nlte_Li_giant_scatt_idlsme.grd']
endfor

print,nltee

; STRUCTURE
sme = { $
   version        :  5.0       , $
   id             :  systime()            , $
   teff           :  teff,$
   grav           :  logg,$
   feh            :  feh,$
   vmic           :  vmic,$
   vmac           :  0.0          , $
   vsini          :  0.0         , $
   vrad           :  0.0          , $
   vrad_flag      :  -2     , $
   cscale         :  1.0        , $
   cscale_flag    :  -3   , $
   gam6           :  1.0          , $
   accwi          :  0.002         , $
   accrt          :  0.0001         , $
   clim           :  0.01          , $
   maxiter        :  20       , $
   chirat         :  0.001        , $
   nmu            :  7           , $
   abund          :  abund         , $
   mu             :  reverse(sqrt(0.5*(2*dindgen(7)+1)/float(7)))            , $
   atmo : { $
      source :  'marcs2014.sav' , $
      method :  'grid', $
      depth  :  'RHOX', $
      interp :  'TAU', $
      geom   :  'PP' $
   }, $
   nlte : { $
      nlte_pro          :  'sme_nlte'    , $
      nlte_elem_flags   : nlte_elem_flags, $
      nlte_subgrid_size : [3,3,3,3]      , $
      nlte_grids        : nlte_grids     , $
      nlte_debug        : 1                $
   }, $
   sob            :  sob           , $
   uob            :  uob           , $
   obs_name       :  outfile      , $
   obs_type       :  3      , $
   iptype         :  'gauss'        , $
   glob_free      :  string(-1)     , $
   ab_free        :  intarr(99)       , $
   gf_free        :  gf_free       , $
   species        :  species       , $
   atomic         :  atomic        , $
   lande          :  lande         , $
   depth          :  depth         , $
   lineref        :  lineref       , $
   line_term_low  :  term_low      , $
   line_term_upp  :  term_upp      , $
   short_format   :  short_format  , $
   line_extra     :  extra         , $
   line_lulande   :  lulande       , $
   nseg           :  nseg          , $
   wran           :  wran          , $
   wave           :  wave          , $
   wind           :  wind          , $
   mob            :  mob           , $
   ipres          :  1e6         , $
   auto_alpha     :  0      $    
  }

save, sme, file=outfile+'.inp'

END

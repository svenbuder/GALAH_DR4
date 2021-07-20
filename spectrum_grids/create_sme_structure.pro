PRO create_sme_structure,index

; PARAMETERS
parameters = mrdfits('marcs2014/marcs2014_scaledsolar_210720.fits',1)

; ABUNDANCES
abundances = mrdfits('marcs2014/marcs2014_scaledsolar_210720.fits',2)
abund = abundances[index]

; WAVELENGTH GRID
nseg = 4
wave1 = 
wave2 = 
wave3 = 
wave4 = 

wran = dblarr(2,nseg)
wran[0,0] = [wave[0]]
wran[1,0] = [wave[wind[0]]]
for i=0,nseg-2 do begin
   wran[0,i+1]=[wave[wind[i]+1]]
   wran[1,i+1]=[wave[wind[i+1]]]
endfor

sob = [[1.+0.*wave1],[1.+0.*wave2],[1.+0.*wave3],[1.+0.*wave4]]
uob = [[0.*wave1],[0.*wave2],[0.*wave3],[0.*wave4]]

; LINELIST
line=mrdfits('galah_master_v5.2.fits',1,/silent)

line_merge, 'LINELIST/'+line_list, line_atomic, line_species, line_lande, line_depth, line_ref,term_low=line_term_low,term_upp=line_term_upp,short_format=short_format,extra=line_extra,lulande=line_lulande

;SELECT ONLY THOSE WITHIN wave1, wave2, wave3, wave4... via j
j=[0]
for i=0,nseg-1 do begin
   k=where(line_atomic[2,*] ge wran[0,i] and line_atomic[2,*] le wran[1,i],kc)
   if kc ne 0 then j=[j,k] 
endfor

nselect   =  n_elements(j)-1
if nselect gt 0 then j=unique(j[1:*])
nselect   =  n_elements(j)

species   =  line_species  [j]
atomic    =  line_atomic [*,j]
lande     =  line_lande    [j]
depth     =  line_depth    [j]
lineref   =  line_ref      [j]
term_low  =  line_term_low [j]
term_upp  =  line_term_upp [j]
extra     =  line_extra  [*,j]
lulande   =  line_lulande[*,j]


; NLTE
nlte_elem_flags = bytarr(99)
nlte_grids = strarr(99)
nltee = ['Li','C','O','Na','Mg','Al','Si','K','Ca','Mn','Ba']
nltez = [  3 , 6 , 8 , 11 , 12,  13 , 14 , 19, 20 , 25 , 56 ]
inlte = where(nltee eq elem) & inlte=inlte[0]
nlte_elem_flags[nltez[inlte]-1]  = 1B
nlte_grids[nltez[inlte]-1]       = ['Amarsi19_']+nltee[inlte]+['.grd']
if nltee[inlte] eq 'Li' and teff lt 5750. then nlte_grids[nltez[inlte]-1] = ['Amarsi19_']+nltee[inlte]+['_t3800_5750.grd']
if nltee[inlte] eq 'Li' and teff ge 5750. then nlte_grids[nltez[inlte]-1] = ['Amarsi19_']+nltee[inlte]+['_t5750_8000.grd']

; STRUCTURE
sme = { $
   version        :  '580'       , $
   id             :  systime()            , $
   teff           :  parameters['TEFF'][index]          , $
   grav           :  parameters['LOGG'][index]          , $
   feh            :  parameters['FEH'][index]           , $
   vmic           :  paramerers['VMIC'][index]          , $
   vmac           :  0.0          , $
   vsini          :  0.0         , $
   vrad           :  0.0          , $
   vrad_flag      :  -1     , $
   cscale         :  1.0        , $
   cscale_flag    :  -3   , $
   gam6           :  1.0          , $
   accwi          :  0.005         , $
   accrt          :  0.005         , $
   clim           :  0.01          , $
   maxiter        :  1       , $
   chirat         :  0.001        , $
   nmu            :  7           , $
   abund          :  abund         , $
   mu             :  reverse(sqrt(0.5*(2*dindgen(nmu)+1)/float(nmu)))            , $
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
   obs_name       :  obs_name      , $
   obs_type       :  obs_type      , $
   iptype         :  iptype        , $
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
   ipres          :  ipres         , $
   auto_alpha     :  0      $    
  }

END

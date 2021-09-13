# coding = utf-42
import numpy as np
import math
import meta_stuff as mt

# die folgende Funktion soll später benutzt werden um durch superposition verschiedene Daten erzeugen
# zu können. Beispielsweise:
#      (Szenario, meta_data) = analytic_data(v_1,5) + analytic_data(v_2,5)
#      # Szenario mit 10 Wellen zweier verschiedener Ausbreitungsgeschwindigkeiten
#      # meta_data gibt zu jedem Zeitpunkt die Information, (v_1:N_v1, v_2:N_v2, ..., v_max:N_vmax).

#@TODO init here!
# def initMeta(I)
#  return mt.Meta(I)

def amp_at(x,t,c,wl,nomS,rt):
  # noms: number of minimal sections; rt: delay rate
  # we need a function that tells us, if a position lies at an inaktive wave..
  # if yes, return 0. if not, we need to refer to the actual wave form
  orig_pos = x - c*t
  temp = orig_pos
  nth_wave = 0
  if (temp < 0):
    while (temp < -(1+rt)*wl):
      temp = temp + (1+rt)*wl
      nth_wave+=1
  if ((orig_pos > 0) or (orig_pos < -wl*nomS) or (temp < -wl)): # is not aktive
    #if (orig_pos < -wl*nomS):
    #  print("Too late, waves are already gone:")
    #  print("  x was {}. Set function value to zero.".format(x_old))
    #  print("  orig_pos: {}".format(orig_pos))
    #  print("  -wl*nomS: {}".format(-wl*nomS))
    #if (temp < -wl):
      #print("This position is in a delay section:")
      #print(" temp = {} < {} = - wave length.".format(temp,-wl))
      #print("  orig_pos: {}".format(orig_pos))
    return 0
  else:
    return (1 - math.cos(2*math.pi*(orig_pos/wl+rt*nth_wave)))/2.0
    
def toArray(n_waves,wave_speed,N_time,N_space,length,dt,delay_rate=0.0,detect_crit=0.5,wave_length_must_be=0):
  # wird nicht wirklich genutzt. beschränke auf iData_
  
  #print("wave_speed={}m/s".format(wave_speed))
  #print("length={}m".format(length))
  #print("dt={}s".format(dt))
  #print("delay_rate={}".format(delay_rate))
  # output:
  # diskrete EMG Signale,
  #   * mit IMPLIZITER Dauer end_time
  #   * Amplitude amp ## braucht man eigentlich nicht, weil man außen einfach sol*amp machen kann.
  #   * bestehend aus n_waves Wellen
  #   * aufgelöst in N_space diskrete Raumpunkte
  #   *           und N_time diskrete Zeitpunkte
  #   * mit Ausbreitungsgeschwindigkeit wave_speed
  #   * 
  # Annahmen:
  #   * Der Muskel ist 7.5 cm lang (0.075m). das sind (16-1)*0.5cm ~ adhesive array, Merletti 2019
  #length = 0.075 #[m]
  #   * die Spannweite dt zwischen den Zeitpunkten beträgt .5ms (0.0005s)
  #     laut Thomas ist eine Abtastrate von 2048 Hz sowas wie der Standard bei Experimenten.
  #dt = 0.0005 #[s]
  # Wird T wirklich benötigt, oder ist es besser, das hier drin zu bestimmen?
  # was hängt da alles mit zusammen?
  #  ... dt
  #  ... frequenz
  #  ... lambda_
  # eigentlich sollte dt fix sein, dass alle Bilder gleich zu deuten sind.
  # wäre es konsistent, wenn man auch DX (=length/(N_space-1)) fix setzt?
  #  ... ich entscheide mich vorerst dagegen, da ich diese Frage lieber als Ergebnis
  #      offen lasse: Wie nah müssen meine Elektroden sein, dass mein Device funktioniert?
  #      --> gibt es so was auf dem Markt, oder muss es erst entwickelt werden?
  #  * hier warte ich auch auf Rückmeldung von Thomas
  #
  # der user will kontrollieren, wie die diskretisierung aussieht.
  # zusätzlich ist die Zeit-diskretisierung eigentlich fix durch die Gerätschaft, also die Auslesefrequenz.
  # da die Anzahl an Zeitschritten, sowie dt fix sind, ergibt sich daraus unwillkürlich die Dauer
  # des Datensatzes.
  # ... ALSO: ergibt sich hieraus direkt end_time:
  end_time = dt * (N_time - 1) # 'minus eins', da N_time die Anzahl an Stellen und nicht die Anzahl an zwischen-Intervallen ist.
  # WIR BRAUCHEN KEINE RANDBEDINGUNGEN FÜR EINE ANALYTISCHE LÖSUNG!
  # Wir haben einen inflow-Rand (Dirichlet-Randbedingungen)
  # Und einen outflow-Rand (mixed formulation?)
  # zum Zeitpunkt t=0 ist alles 0.
  # 
  # Wir können hierzu schlichtweg die sin()-Funktion verwenden...
  # mit der Ausbreitungsgeschwindigkeit bestimmt man, ob die erste welle schon da war, und die letzte noch kommt, ansonsten null. Wenn ja, rutscht der Punkt durch das Intervall [0,n_waves*2\pi] oder ähnlich.
  # man könnte als Welle f(x) = 1 - cos (x), x \in [0,2\pi] benutzen.
  #
  #
  # wir setzen etwas fest: die Wellenbreite, soll vom user vorgegeben werden: NEIN ! ! !
  # lambda_ = wavelength #[m] (= c/f). Hier könnte ein Problem auftreten... ausprobieren.
  # ACHTUNG: Wenn der user die breite vorgibt, dann funktioniert das nur, wenn sie sich überlagern.
  # das ist physikalisch auch sinnvoll: zwei nahe gelegene Fasern zur selben Zeit aktiviert,
  # ergeben zwei sich überlagernde Wellen.. Die Frage ist, ob das in diese funktion muss.
  # es wäre vermutlich sinnvoller, das von außen zu kontrollieren um gewollt solche szenarien zu erzeugen.
  # wenn wir die überlagerung also zunächst nicht zulassen, dann können wir hier frei wählen.
  # Langsam wird das aber so komplex, dass ich mich frage, ob man nicht gleich EMG-Daten in opendihu erzeugt,
  # und diese verwendet. Dann brauche ich mir nicht Gedanken darüber machen, ob die Ergebnisse überhaupt
  # übertragbar sind.
  #
  #
  #
  # unsere Anregung soll grob so aussehen:._/\_._____._/\_._____._
  # sie besteht also aus einer ganzen welle '_/\_' der Länge lambda_, plus einem delay '_____' der Länge delay_.
  # zu Beginn gibt es keinen Delay. Es beginnt mit f(0).
  # spätestens jetzt müssen wir f(x) von [0,2pi] auf ein anderes Intervall ziehen.
  # wir haben c = wave_speed gegeben und kennen lambda_. Daraus ergibt sich f = c / lambda_.
  # Das führt wiederum zu der Periodendauer T = 1 / f = lambda_/c.
  # Also lebt f auf [0,T].
  # da wir aber keine Numerik betreiben hier, sondern analytisch die Welle berechnen,
  # brauchen wir einfach nur eine Funktion Ausschlag(x,t) ! ! ! !
  # Dazu setzen wir also den Positions-vektor:
  pos = np.linspace(0,length,N_space)
  sol = np.zeros(N_space*N_time)
  sol.shape = (N_time,N_space)
  #time = np.linspace(0,end_time,N_time)
  # wir haben also die Aufgabe n_waves mit der geschwindigkeit wave_speed innerhalb einer Zeit end_time
  # über den Muskel zu ballern. Dabei sind gewisse Bedingungen einzuhalten, es gibt aber auch Freiheiten
  # wir können schmale wellen gemeinsam haben, oder breite wellen eher nacheinander.
  # egal, was man tut, man kann von außen immernoch überlagern. welche Freiheit, gibt uns also mehrwert?
  # was man z.B. machen könnte, ist, dass man eine Rate à la Pause zu Periodenlänge nimmt:
  #  - 0: keine Pause zwischen den Wellen = rein trigonometrische Funktion.
  #  - 1: zwischen jeder Welle ist eine Welle "abgeschaltet" 
  #  - 2: zwischen jeder Welle ist eine Pause der Länge von zwei Wellen.
  #  - etc.
  # Dieses Feature habe ich jetz mal delay_rate genannt.
  #
  

  # Jetzt ist also die Frage, wie man das alles unter einen Hut bekommt. Vielleicht programmiert man einfach mal los..
  # um außen eine einfache handhabung zu erreichen, soll n_waves heißen,
  # dass zu mindestens einem Zeitpunkt n_waves Wellen gleichzeitig und vollständig vorhanden sind.
  # Das heißt, ich berechne aus n_waves, length und delay_rate zunächst mal die Wellenlänge
  # allerdings will ich verhindern, dass es eine Wellenlänge gibt, die größer als 6 cm ist.
  # Das bekomm ich hin indem ich bei n_waves=1 und gleichzietig delay_rate=0 die randomly_more_sections
  # auf mindestens 1/3 setze. --> lambda_ = 0.08 / (1.333) = 0.06.
  number_of_minimal_sections = n_waves + (n_waves-1)*delay_rate
  randomly_more_sections = 0
  if wave_length_must_be == 0: ###########
    # weil ich nicht will, dass sich hier irgendein randeffekt einschleicht:
    if (delay_rate!=0):
      randomly_more_sections = np.random.choice(math.ceil(delay_rate), 1)[0] # this takes one of {0, 1, .. delay_rate}.
    if (length / (number_of_minimal_sections+randomly_more_sections) > 0.06):
      print("limiting ...")
      randomly_more_sections = (length-0.06)/0.06
    #else:
    #  randomly_more_sections=0  
    lambda_ = length / (number_of_minimal_sections+randomly_more_sections)
    if lambda_>0.06:
      print("Something went wrong in limiting the wave length. ({}cm > 6cm)".format(lambda_*100))
  else:
    lambda_ = wave_length_must_be
  # warn user if we have aliasing effects:
  if (lambda_<pos[2]):# weil pos[2] = 2*DX
    if (2*lambda_<pos[2]):
      if (10*lambda_ < pos[2]):
        print("SERIOUS ALIASING: wave length {0:0.1E}cm is MUCH smaller than 2*IED = {1:0.1E}cm!".format(lambda_*100,100*pos[2]))
      else:
        print("WARNING: wave length {0:0.1E}cm is smaller than 2*IED = {1:0.1E}cm!".format(lambda_*100,pos[2]*100))
    else:
      print("Warning: wave length {0:0.2f}cm is smaller than 2*IED = {1:0.2f}cm.".format(lambda_*100,pos[2]*100))
  # in diesem Vorgehen allein, ergibt sich ein Problem:
  # ist die Geschwindigkeit zu klein, kann es passieren, dass die Wellen noch gar nicht alle
  # nach innen propagiert sind, die Zeit end_time aber bereits erreicht ist. In diesem Fall
  #  ..muss ich entweder (a) die Wellen stauchen, so dass sie alle vollständig sind (wenn auch am Rand)
  #    oder (b) ich erzeuge eine Warnung / einen Fehler.
  # Mir wäre (a) lieber, sofern ich das irgendwie 'smooth' einbauen kann.
  # Wenn (b) sein muss, dann wäre hier hilfreich, dem User auch zu sagen,
  # wie er die Parameter anpassen muss, dass es funktioniert.
  # am einfachsten wäre hier auszugeben, wie hoch die Wellengeschwindigkeit mindestens sein muss,
  # oder wie welche Diskretisierung N_time notwendig ist.
  # DETEKTIERE PROBLEM:
  dist = wave_speed * end_time # so weit kommt die Front innerhalb der bereitgestellten Zeit.
  if (dist < lambda_* number_of_minimal_sections): # hier lassen wir die randoms weg. (wenn durch, dann durch.)
    print("Wave speed = {}m/s:".format(wave_speed))
    print("  In analytic_data(): Waves too slow for all ot them to propagate onto domain.")
    #print("  dist = {}".format(dist))
    min_w_sp = number_of_minimal_sections*lambda_ / end_time
    min_N_tm = math.ceil(lambda_* number_of_minimal_sections / wave_speed / dt)
    print("  Try with a higher wave speed of >= {}m/s or use at least {} time discretization points.".format(min_w_sp,min_N_tm))
    
  if (dist < length + lambda_*(number_of_minimal_sections+randomly_more_sections)):
    faster = (length + lambda_*(number_of_minimal_sections+randomly_more_sections))/ end_time
    more = math.ceil((length + lambda_*(number_of_minimal_sections+randomly_more_sections)) / wave_speed / dt)
    print("Wave speed = {}m/s:".format(wave_speed))
    print("  No smooth end of this section, since not all waves left the domain.")
    print("  Could be guaranteed with a speed of >= {}m/s.".format(faster))
    print("  ..likewise you can simply use at least {} time discretization points.".format(more))
  #  gleichermaßen kann die Geschwindigkeit natürlich sehr hoch sein. Dann ist die Frage, ob man
  #  die Wellen einfach auslaufen lässt (und für einen beträchtlichen Zeitraum einfach Nullen erhält),
  #  oder ob man weiterhin Wellen erzeugt bis kurz vor Ende.
  #  der Einfachheit würde ich einfach auslaufen lassen. Da
  
  # else: ####### wave_length wurde von außen vorgegeben, dann ist number_of_minimal_sections gleich wie
  #            #  im andern fall, delay_rate auch
  
  # kann deutlich schneller berechnen, wenn delay_rate = 0
  # if delay_rate != 0:
  for i in range(N_time):
    for j in range(N_space):
      sol[i][j] = amp_at(pos[j],i*dt,wave_speed,lambda_,number_of_minimal_sections,delay_rate)
      
  # meta daten:
  #
  # Für jedes t können wir bestimmen, wie viele Wellen im Gebiet sind.
  # allerdings ist dies abhängig davon, welche Teilwellen wir als "vorhanden" zählen lassen.
  # wir könnten das vom user bestimmen lassen. Beispielsweise könnte eine Welle
  # da sein, wenn mindestens 50% ihres Trägers im Gebiet sind.
  # wir nutzen detect_crit (default: 0.5).
  # Ausbreitungsgeschwindigkeit ist wave_speed.
  
  # Object to hold the data
  meta = mt.Meta(N_time)
  #
  time_when_first_is_in = detect_crit*lambda_/wave_speed
  #time_when_last_is_in = (number_of_minimal_sections-detect_crit)*lambda_/wave_speed
  time_for_one_to_enter = (1+delay_rate)*lambda_/wave_speed
  time_when_first_leaves = (length+detect_crit*lambda_)/wave_speed
  #print("wave_speed: {}".format(wave_speed))
  #print("  first leaves at t = {}s".format(time_when_first_leaves))
  for i in range(N_time):
    n_waves_came_in=0
    n_waves_got_out=0
    #print("i={}".format(i))
    if (i*dt >= time_when_first_is_in):
      t_d = i*dt - time_when_first_is_in
      n_waves_came_in = 1 + min(int(t_d / time_for_one_to_enter),n_waves-1)
      #print("  adding {}".format(n_waves_came_in))
    if (i*dt>=time_when_first_leaves):
      t_d = i*dt - time_when_first_leaves
      n_waves_got_out = 1 + min(n_waves-1,int(t_d / time_for_one_to_enter)) # hier wird abgerundet. daher brauchen wir +1
      #print("  minus  {}".format(n_waves_got_out))
    meta.iData_[i][wave_speed] = n_waves_came_in - n_waves_got_out
  
  #for i in range(N_time):
  #  print("step = {}".format(i+1))
  #  if meta.data_[i][wave_speed]!=1:
  #    print("  Detected {} waves.".format(meta.data_[i][wave_speed]))
  #  else:
  #    print("  Detected 1 wave.")
  
  return sol,meta

def is_in_gap_or_off_completely(X,zero_intervals,lambda_):
  #print("zero intervals: {}".format(zero_intervals))
  #print("X = {}".format(-X))
  # X in the beginning is >=0, but than usually < 0.
  # zero_intervals are on the positive half ray
  if X > 0:
    #print("Is OFF")
    #return False
    #off completely:
    return True
  if (X < -(zero_intervals[-1,1]+lambda_)):
    #return False # wäre richtig wenn nicht off und wenn nicht l eingerechnet wäre
    #off completely:
    #print("Is OFF")
    return True
  else:
    promissing_interval_indx = np.searchsorted(zero_intervals[:,1],-X)
    if (promissing_interval_indx==zero_intervals.shape[0]):
      # -X is larger than zero_intervals[-1,1] by less then lambda_ => it is in the last wave
      return False
    else:
      # test whether -X is in interval
      if (zero_intervals[promissing_interval_indx,0] <= -X and -X <= zero_intervals[promissing_interval_indx,1]): # X ist im Intervall
        return True
      else:
        #print("In WAVE")
        return False
 
def in_wave_no(position,Zintervals,lambda_):
  ######################
  # returns 
  #   * 'wave_number,True' of the wave the position belongs to, if it is inside a wave, or at its border, 
  #   * 'gap_number,False' of the  gap the position belongs to, if it is in the interior of a gap,
  #   * '-1,False'         if the position is before the first wave,
  #   * 'W,False'          if the position lays after the last wave.
  # Here, we have G gaps and W = G+1 waves.
  # let us name the waves w_0, w_1, ... w_G, note that |(w_i)| = W
  # let us name the gaps  g_1, g_2, ... g_G = g_{W-1} INSIDE this function
  # (OUTSIDE of this function we use w_0,...,w_G and g_0,...g_{ G-1 = W-2 } !!!)
  ######################
  # check one thing before doing anything:
  if (position > 0):
    # position before first wave
    # Outside we expect information, that we are before the first gap and not in a wave:
    #print("returning {},{}".format(-1,False))
    ret_value=[-1,False]
    return ret_value
  
  pos = -position
  # next line will return the index of the (wave number OR the gap number) we are in
  no = np.searchsorted(Zintervals[:,0],pos)
  inWave = True
  indicator = np.searchsorted(Zintervals[:,1],pos)
  if (indicator != no):
    # we are in the interior or at the right boundary (when speaking of pos in [0.1,0.4]) border of a gap.
    # (Thats the LEFT boundary when speaking of position in [-0.4,-0.1]) 
    if (Zintervals[indicator,1] != pos):
      # we are in the interior 
      inWave = False
    #else:
    #  # we are at the boundary, which counts as wave.
    #  #print("no correction necessary")
  if (pos > lambda_ + Zintervals[-1][1]):
    # pos is after the last wave
    # no is already G+1, but now we need to set inWave = False which means, 'no' will be decreased before return,
    # which would make it wrong, so increase it by one:
    no +=1
    inWave = False
  # as a last step we need to convert the number no from INSIDE meaning to OUTSIDE meaning, but only for gaps:
  #print("returning {},{}".format((no - (not inWave)),inWave))
  ret_value = [(no - (not inWave)),inWave]
  return ret_value

def get_number_of_waves(shft,zero_interv,leng,lamb,factor=0.5):
  #                                                 a-A
  #  IN: _Axxaxx_ = _xAxxax_ = _xxAxxa_ True,True   0 diff
  #  Out: _xxxAx_a = _xxxxA__a          True,False  0 diff
  #  Out: _xxxxA_axxx__                 True,True  -1 diff
  #
  #                                                 b-B
  #  Out: _b_xBxx = _b_Bxxxx_           False,True  0 diff
  #  IN: _bxxBxx_ = _xbxxBx_ = _xxbxxB_ True,True   0 diff
  #  Out: xb_xBxx_                      True,True   1 diff
  #
  A, B = shft, leng+shft
  res_A = in_wave_no(A,zero_interv,lamb)
  res_B = in_wave_no(B,zero_interv,lamb)
  waves = res_A[0]-res_B[0]-1
  #print(waves)
  if res_A[1]:
    a = in_wave_no(A+lamb*factor,zero_interv,lamb)
    if (a[1] and a[0]==res_A[0]):
      #print("A")
      waves += 1
  else:
    waves += 1
  if res_B[1]:
    b = in_wave_no(B-lamb*factor,zero_interv,lamb)
    if (b[1] and b[0]==res_B[0]):
      #print("B")
      waves += 1
  return waves
  
def get_probabilistic_number_of_waves(shift,zero_interv,leng,lambd,criterion):
  #uu=-0.07
  #oo=-0.097
  #print(leng)
  #print(lambd)
  #print(criterion)
  # anstatt, dass die Funktion N zurück gibt, soll eine Art Vektor zurück gegeben werden:
  # Zum Beispiel in der Situation:
  # __|__^__^_^_^___^___|_ sind ohne Frage 5 Wellen im Gebiet. --> return {5: 1.0}
  # wenn aber 
  # __|__^__^_^_^___^__^|^^^_
  # dann haben wir fünf ganze und zusätlich eine viertel Welle drin. Wie soll das behandelt werden?
  # Bisher war eine Welle IN, wenn mindestens 0.5 ihrer Länge im Gebiet war. bei diesem Wert sind wir 
  # also von 5 auf 6 gesprungen. genau sechs ist aber, wenn die sechste zu 100% im Gebiet ist. Das heißt
  # eigentlich wäre der "richtige" Wert beim Sprung 5.5 und nicht 6. entsprechend wäre also obige Situation
  # als 5.25 zu werten. also return {5: 0.75, 6: 0.25}
  # Andererseits könnte man asu Sicht des menschlichen Betrachters auch sagen, dass die sechste Welle 
  # "da ist", sobald ein gewisser Threashold erreicht wurde.. Beispielsweise 1/3 der Amplitude. (entspricht 0,26772 (= arccos(1-0.3333333333333)/pi) der wellenlänge)
  # ist weniger orhanden, will ich aber nihct eine Stufe, sondern einen stetigen Übergang zwischen den beiden Zuständen.
  #
  # 5 --> 5.0
  # 5.1 --> 5.3735
  # 5.2 --> 5.747
  # 5.26772 --> 6.0
  # 5.3 --> 6.0
  ##
  # ebenso ist der linke Rand zu handhaben.
  # _^|^^__^___^__|_ --> return {2: 0.33, 3: 0.66}
  #
  # haben wir zwei Wellen am Rand, dann kommt womöglich sowas raus wie {0:30, 1:40, 2:30} das heißt es gibt drei Zustände,
  # die inneinander übergehen. Wie wird sowas behandelt?
  #
  # bei zwei möglichkeiten ist eine Deutung von zB. '5.5' noch eindeutig: Es ist gleichbedeutend mit {5:0.5, 6:0.5}.
  # bei drei möglickeiten, könnte '5.5' aber ENTWEDER heißen: {4:0.1, 5:0.3, 6:0.6}, ODER eben {4:0.083, 5:0.333, 6:0.583}.
  # (In beiden Fällen gilt, dass P_min nicht innen liegen darf! außerdem ist Sum(P_i) = 1)
  # Das ist ein Problem, da die zwei Methoden irgendwo kompatibel sein sollten..
  
  ######
  # step 1: get number of inner waves (those that are not touching the boundary)
  # factor = 0.0 will count waves that are completely in the (closed) domain: wave-boundaries might coincide with the domain boundaries.
  N_inner_waves = get_number_of_waves(shift,zero_interv,leng,lambd,factor = 1.0)

  #####
  # step 2: get the fractions of waves that intersect with the boundaries
  #
  # set default values
  FA = 0
  FB = 0
  # check whether the boundaries lay in a wave / gap
  A,B = shift, leng+shift # A is the point in the reference domain that was 0 before transformation. Call this one Front Domain Boundary
  (BoundaryAIsAtWaveN, NumberIsWave_A) =  in_wave_no(A,zero_interv,lambd)
  (BoundaryBIsAtWaveN, NumberIsWave_B) =  in_wave_no(B,zero_interv,lambd)
  #if (shift > oo and shift < uu):
    #print()
  #  print("-A = "+str(-A))
  #  print("-B = "+str(-B))
  #  print(zero_interv)
  # if not in gap, we can compute the fractions
  waveBoundaryLeft = 0.0
  waveBoundaryRight = 0.0
  if NumberIsWave_A:
    if BoundaryAIsAtWaveN < zero_interv.shape[0]:
      # for the wave at the front domain boundary side, waveBoundaryLeft is the wave boundary inside the domain
      waveBoundaryLeft = zero_interv[BoundaryAIsAtWaveN][0]-lambd
      # .. and waveBuondaryRight is the wave boundary outside the domain (travelling nearer)
      waveBoundaryRight = zero_interv[BoundaryAIsAtWaveN][0]
    else:
      waveBoundaryLeft = zero_interv[BoundaryAIsAtWaveN-1][1]
      waveBoundaryRight = zero_interv[BoundaryAIsAtWaveN-1][1]+lambd
    # we need to make sure that this wave is not contained in the domain completely:
    if waveBoundaryRight != -A:
      #wave_length = waveBoundaryRight - waveBoundaryLeft
      FA = (-A-waveBoundaryLeft)/lambd #wave_length
   #   if (shift > oo and shift < uu):
   #     print("FA = {}".format(FA))
      if FA >= criterion:
        N_inner_waves += 1
  #      if (shift > oo and shift < uu):
  #        print("1 added since FA > C")
        FA = 0
      if (FA<0 or FA==1.0):
  #      if (shift > oo and shift < uu):
          print("Fehler in get_probabilistic_number_of_waves()!")

  if NumberIsWave_B:
  # es ist sicher gestellt, dass am Puntk B eine Welle ist.
  #  Fall 1: ist es die erstkante, so ist die welle komplett drin.
  #  Fall 2: ist es die schlusskante, so ist die welle quasi draußen, heißt: FB = 0 (Fall ist auch in Fall 3 enthalten)
  #  Fall 3: ist es zwischendrin, so ist zu ermitteln, wie groß der Anteil innerhalb ist FB = ?
  #  Bildchen:
  #   Falsch: wenn -B<0, dann wären die Wellen noch nicht bei B:  |-B(=-0.049)<...    [W0Front >-A(=0.001)| W0Back ... ]
  #   Richtig: -B>=0:  [W0Front |-B(=0.005)< W0Back=G0Front____G0BACK=W1FRONT ... ] ...  >-A(=0.055)|
  #   hier sind die wichtigen Infos -B==W0FRONT (Überprüfung von Fall 1) und 'W0BACK - (-B)' (Ermittlung FB, Fall 2&3)
  #   haben wir fall 1, dann ist die welle bereits gezählt. es ist nichts zu tun
  #   haben wir fall 2/3, dann müssen wir [N++, solange FB>C] oder [P(..) wenn 0<=FB<C]
  
  
    if BoundaryAIsAtWaveN < zero_interv.shape[0]:
      # for the wave at the back domain boundary side, waveBoundaryLeft is the wave boundary outside the domain
      #.. das entspricht WXFRONT
      waveBoundaryLeft = zero_interv[BoundaryBIsAtWaveN][0]-lambd
      # .. and waveBoundaryRight is the wave boundary inside the domain (travelling nearer)
      #.. das entspricht WXBACK
      waveBoundaryRight = zero_interv[BoundaryBIsAtWaveN][0]
    else:
      waveBoundaryLeft = zero_interv[BoundaryAIsAtWaveN-1][1]
      waveBoundaryRight = zero_interv[BoundaryAIsAtWaveN-1][1]+lambd
    # we need to make sure that this wave is not contained in the domain completely:
    if waveBoundaryLeft != -B:
      #wave_length = waveBoundaryRight - waveBoundaryLeft
      FB = (waveBoundaryRight + B)/lambd #wave_length Note that '+B' = '-(-B)'
   #   if (shift > oo and shift < uu):
   #     print("FB = {}".format(FB))
      if FB > criterion:
        N_inner_waves += 1
   #     if (shift > oo and shift < uu):
   #       print("1 added since FB > C")
        FB = 0
      if (FB<0 or FB==1.0):
   #     if (shift > oo and shift < uu):
          print("Fehler in get_probabilistic_number_of_waves()!")
  
  #####
  # step 3: get the probabilities P{N},P{N+1} and P{N+2}
  # wie bekomme ich die Wahrscheinlichkeiten P_{so viel sind auf jeden Fall drin = i0}, P_{i0+1} und P{i0+2} berechnet?
  #
  # Illustratives Beispiel und Herleitung:
  # Es sei bekannt, dass (o.B.d.A.) der Anteil der linken Welle, FL = 0.1, der Anteil der rechten Welle sei FR = 0.15 und
  # die Anzahl an ganzen Wellen im Inneren sei NW. Dann ist gesucht:
  #                                   P_{NW}/{N}, P_{NW+1}/{1}   und   P{NW+2}/{2}.
  # Außerdem fordern wir
  #                            SUM(P_i) = 1.0    und    P_{NW+1} != min arg(P_i).
  # Wir wollen außerdem erreichen, dass wenn FL xoder FR = 0, der lineare Übergang zwischen den Übergängen reproduziert wird 
  # und wenn FL=FR=0, dass der triviale Fall P_NW=1 enthalten ist.
  # PL ist die Wahrscheinlichkeit, dass die linke Welle zählt.
  #                                  PL := FL/C,   wobei zum Beispiel   C = 0.26772.
  # Ebenso ist PR = FR/C.
  # PN ist die Wahrscheinlichkeit, dass keine der beiden anderen Wellen zählt:
  #                               PN = (1-PL)*(1-PR).
  # P2 ist die Wahrscheinlichkeit, dass beide der anderen Wellen zählen:
  #                               P2 = PL*PR
  # P1 ist die Wahrscheinlichkeit, dass genau eine der beiden Wellen zählt:
  #                               P1 = 1 - PN - P2
  #
  ## Die Funktion (PN,P1,P2) = P(FL,FR,C) tut genau das ::
  #
  def P(FL,FR,C):
    # wahrscheinlichkeitsberechnung mit eingebauter Wertbegrenzung
    # modell basiert auf linearem Übergang bei eingehender Welle in Bezug auf die wellenbreite
    # (das ist so in der Erzeugung. Wie das NN es handhabt ist unklar!)
    PL = min(FL,C)/C
    PR = min(FR,C)/C
    P2 = PL*PR
    PN = (1-PL)*(1-PR)
    P1 = 1 - PN - P2
    return (PN,P1,P2)
 
  P_values = P(FA,FB,criterion)

  ret_value={}
  ret_value[N_inner_waves] = P_values[0]
  for el in P_values[1:]:
    if el != 0:
      ret_value[int(N_inner_waves+len(ret_value))] = el
  #if (shift > oo and shift < uu):
  #  print(ret_value)

  return ret_value

def appendScenario(v,l,gap_vec,N_space,length,dt,crit):
  # similar to accumulate, but with variable distances between waves.
  # zero_intervals = [I_0,...,I_{-1}] holds the Intervals I_i=[left,right] where there is no sEMG signal. (as a reference at t=0)
  # In between these I_i, there are waves of length l.
  # 'length' is the length of the adhesive array (that's the electroids holding device)
  #print("N_space={}".format(N_space))
  zero_intervals = np.zeros((len(gap_vec),2))
  # zero_intervals looks like [[0.1,0.2],[0.3,0.6], ..., [2.6,6.7], [6.8,7.1]], > 0
  # we speak of left and right boundaries.
  sum_of_gaps_before = [0]
  for ith_gap in range(len(gap_vec)):
    # 0th gap is AFTER the first wave.
    # last gap is BEFORE the last wave.
    passed_waves = ith_gap + 1
    left = passed_waves*l + sum_of_gaps_before[ith_gap]
    right= left + gap_vec[ith_gap]
    zero_intervals[ith_gap]=[left,right]
    sum_of_gaps_before.append(sum_of_gaps_before[-1] + gap_vec[ith_gap])  # > 0
  # über l, v und die gaps können wir abschätzen, wie lange das ganze ding geht..
  # x = forbidden_intervals[-1][1] ist die rechte Intervallgrenze der letzten gap.das heißt 
  # Dist = x + l ist die gesamt-Distanz über die sich die Wellen verteilen.
  # Dist = l + zero_intervals[-1][1]
  # zusätzlich muss aber auch noch die ganze Domain durchlaufen werden
  dist_all = l + zero_intervals[-1][1] + length
  #print("dist_all: {}".format(dist_all))
  # N_time, die Anzahl an Zeit-Auswertungspunkten ergibt sich dann als
  N_time = math.ceil(dist_all/v/dt) + 1 # ist die Anzahl an Zeitschritten plus eins
  #print("N_time={}".format(N_time))
  #end_time = dt * (N_time - 1) # die numerische End-Zeit ist ein Vielfaches von dt und nicht zwingend gleich 'D/v'
  pos = np.linspace(0,length,N_space)
  #print("pos: {}".format(pos))
  # Object to hold the sEMG data
  #print(N_space)
  #print(N_time)
  sol = np.zeros(N_space*N_time)
  sol.shape = (N_time,N_space)

  # warn user if we have aliasing effects:
  if (l < pos[2]):# weil pos[2] = 2*DX
    if (l < pos[2]/2):
      if (l < pos[2]/10):
        print("SERIOUS ALIASING: wave length {0:0.1E}cm is MUCH smaller than 2*IED = {1:0.1E}cm!".format(l*100,100*pos[2]))
      else:
        print("WARNING: wave length {0:0.1E}cm is smaller than 2*IED = {1:0.1E}cm!".format(l*100,pos[2]*100))
    else:
      print("Warning: wave length {0:0.2f}cm is smaller than 2*IED = {1:0.2f}cm.".format(l*100,pos[2]*100))
    
  # Object to hold the meta data
  meta = mt.Meta(N_time)
  #print("meta size: {}".format(meta.size_))
  #
  for i in range(N_time):
    shift = -i*dt*v
    # concerning meta:
    # we know the number W of waves in the system. Using the G=W-1 gap locations, we can determine
    # the first and last wave that are completly inside the domain D = [0,length].
    # let us name the waves w_0, w_1, ... w_G, note that |(w_i)| = W
    # -1st gap is before the first wave
    # (last+1)th gap is after the last wave
    
    # integer meta data:
    n_waves = get_number_of_waves(shift,zero_intervals,length,l)
    #print("wave number at {}*dt: ".format(i,n_waves))
    meta.iData_[i][v] = n_waves
      #print("Tadaa...")
      #print("meta size: {}".format(meta.size_))
      # xxx___xxx____xxx____xxx__xxx_xxx_xxx|  | -1,no 0,yes  0 0
      # xxx___xxx____xxx____xxx__xxx_xxx_x|xx|   0,yes 0,yes  1 0
      # xxx___xxx____xxx____x|xx__xxx_xxx_xxx|   0,yes 3,yes  4 2
      # xxx___xxx____xxx__|__xxx__xxx_xxx_xxx|   0,yes 3,no   4 3
      # xxx___xxx____xxx__|__xxx__xxx_xxx_xx|x   0,yes 3,no   4 3
      # xxx___xxx____xxx|____xxx__xxx_xxx_xx|x   0,yes 4,yes  4 3
      # xxx___xxx____xxx____|xxx__xxx_xxx_xx|x   0,yes 3,yes  4 2
      # xxx___xxx____xxx____|xxx__xxx_xxx_|xxx   0,yes 3,yes  3 2
      # xxxx___xxxAx_a_xxxx__xxxbx_B_xxxx        0,no  3,yes  2 2
      # xxxAx_a_xxxx__xxxx_b_B
      # _A_xaxxx__xxxx_b_B
      # testing:
      # xxA(0)_a(-1)_b(-1)_B(-1)                               0 ind 0
      # xA(0)xxa(0)x___b(-1)_B(-1)                             1 ind 1
      # xA(1)_a(0)_xxxx__b(-1)_B(-1)                           1 ind 1
      # xxxx_A(1)_a(1)xxxx__xxxx___b(-1)_B(-1)                 2 ind 2
      # xA(2)x_a(1)_xxxx____xxxx_b(-1)_B(-1)                   2 ind 2
      # _A(2)_xa(2)xxx__xxxx__xb(0)xxB(0)x                     3 ind 3
      # xA3x_a2_xxxx_xxxxb1_B0_xxxx                            2 ind 2
      # _A3_a3xxxx__xxxx_b1_xB1xxx                             2 ind 3
      # testing more:
      # _A3_a3_xx_xb2x_B1_                                     2 ind 2
      # A3_a3_xxx_b2_xB2x                                      1 ind 1
      # A3_a3_xxxb3x_B2_x                                      1 ind 1
      # A3_a3_xb3xxB3_                                         1 ind 1
      # A3_a3_b3_B3_xx                                         0 ind 0
    # probabilistc meta data:
      #print(i)
    wave_dict = get_probabilistic_number_of_waves(shift,zero_intervals,length,l,crit)
    meta.pData_[i][v] = wave_dict
      #print()  
    #boundaries = int(res_A[1] and res_A_minor[1])+int(res_B[1] and res_B_minor[1])
    #print("guessing {}, with boundaries = {}".format(res_A[0]-max(0,res_B[0])+boundaries,boundaries))
    #print("computed {}".format(n_waves))
    # concerning sol:
    # for a position x in [0, length], the position at the 0-reference is X - i*dt*v. So X can be negative, quite easily
    # X is in a gap => f(x,i*dt) = 0
    # X isnt in gap => f(x,i*dt) = 1-cos(2*pi*X/l)
    for j in range(N_space):
      #print("position: {}".format(pos[j]+shift))
      if is_in_gap_or_off_completely(pos[j]+shift,zero_intervals,l):
        #print("IS in gap.")
        sol[i][j] = 0.0
        #print("|")
      else:
        #print("is NOT in gap.")
        #print("{}\n".format(zero_intervals))
        X = -(pos[j]+shift)
        a,b = 0,0
        #print("X={}".format(X))
        # since X is not in a gap and not on a border, we need to know the border positions of the neighbouring gaps.
        higher_indx = np.searchsorted(zero_intervals[:,0],X)
        #print(higher_indx)
        if higher_indx==0:
          # X is in the first wave between [-l and 0] convert this to ~~> minus_X in [0,l] = [a,b]
          a,b = 0,l
        elif higher_indx==len(zero_intervals):
          # we are in the last wave.
          a,b = zero_intervals[-1,1],(zero_intervals[-1,1]+l) # yes, b > a
        else:
          # we are in an inner wave
          lower_indx = higher_indx - 1
          a,b = zero_intervals[lower_indx,1], zero_intervals[higher_indx,0]
          if(not math.isclose(b-a,l)):
            print("Controle in analytic_data, appendScenario: Wave length problem detected: b-a != l: {} != {}".format(b-a,l))
            
        sol[i][j] = (1-math.cos(2*math.pi*(X-a)/l))/2.0
        #print("a={}, b={} \n".format(a,b))
        #print("f({}) = {}".format(X-a,sol[i][j]))
        #print()
        #print()
        #print()
        #print()        
        #if (sol[i][j] >= 0):
        #  print(sol[i][j])

  return sol,meta








  
import tkinter as tk
from tkinter import ttk
import matplotlib
import re
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sin, cos, exp, pi
import cmath
from scipy.stats import skew, kurtosis, variation
import numpy as np
from matplotlib import rc
import random

#rc('text', usetex=True)

def digits_entry_check(action, index, value_if_allowed, prior_value, text, validation_type, trigger_type, widget_name):
    if action == '1':
        p = re.compile('[\-]?[\d]*\.?[\d]*')
        if p.match(value_if_allowed).end() == len(value_if_allowed):
                try:
                    float(value_if_allowed)
                    return True
                except ValueError:
                    if value_if_allowed[0] == '-' and len(value_if_allowed) == 1:
                        return True
                    return False
        else:
            return False

    return True

class AmplitudeWindow(tk.Toplevel):
    def __init__(self, data, **kwargs):
        self.data = data
        self.linear = True
        tk.Toplevel.__init__(self, **kwargs)
        scaleButton = ttk.Button(self, text="Scale", width = 10, command=self.scaleButtonCallback)
        scaleButton.pack(side=tk.LEFT)
        self.title('Amplitude')
        self.fig = Figure()
        self.fig.suptitle('Amplitude')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plot = self.fig.add_subplot(111)
        self.invalidate()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()


    def invalidate(self):
        self.plot.clear()
        if self.linear:
            self.plot.plot(self.data[0], self.data[1])
        else:
            self.plot.semilogy(self.data[0], self.data[1])
        self.canvas.draw()
        pass

    def scaleButtonCallback(self):
        self.linear = not self.linear
        self.invalidate()


class BaseSignal:
    def __init__(self, parent, controller):
        self.parent = parent
        self.controller = controller
        self.vcmd = (self.parent.register(digits_entry_check), "%d", "%i", "%P", "%s", "%S", "%v", "%V", "%V")
        self.t = 128
        self.deltaT = 1.0
        self.linear = True

    def initInterface(self, properties):
        self.mainFrame = ttk.Frame(self.parent)
        self.mainFrame.grid(row=0, column=0, sticky="news") #??
        self.propertyFrame = ttk.Frame(self.mainFrame, width=160, height=100)
        self.statisticFrame = ttk.Frame(self.mainFrame, width=160, height=100)
        labels = {}
        self.entries = {}
        c = 0
        for k, i in properties.items():
            labels[k] = ttk.Label(self.propertyFrame, text=k)
            labels[k].grid(row=c, column=0)
            self.entries[k] = ttk.Entry(self.propertyFrame, validate="key", validatecommand=self.vcmd, width=7)
            self.entries[k].insert(0, str(i))
            self.entries[k].grid(row=c, column=1)
            c += 1

        setButton = ttk.Button(self.propertyFrame, text="Set", width=7, command=self.setCallback)
        setButton.grid(row=len(labels), column=1, sticky="WE")
        fourierButton = ttk.Button(self.propertyFrame, text="Fourier", width = 10, command=self.show_ft)
        fourierButton.grid(row=len(labels) + 1, column=1, sticky="WE")
        self.propertyFrame.pack(side=tk.LEFT)
        self.statisticFrame.pack(side=tk.RIGHT)
        self.statisticLabels = {}

    def basePlot(self):
        self.figure = Figure(figsize=(5,5), dpi=100) #check settings
        self.canvas = FigureCanvasTkAgg(self.figure, self.mainFrame)
        self.plot = self.figure.add_subplot(111)
        self.figure.suptitle(self.title)
        self.invalidatePlot()
        #self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


        toolbar = NavigationToolbar2TkAgg(self.canvas, self.mainFrame)
        toolbar.update()

    def invalidatePlot(self):
        self.plot.clear()
        #self.statisticFrame.clear()
        self.x = [ i * self.deltaT for i in range(int(self.t)) ]
        self.y = [ self.signal(i) for i in self.x ]

        if self.linear or len(y) > 100:
            self.plot.plot(self.x, self.y)
        else:
            self.plot.stem(self.x, self.y, linefmt='b--'),

        for k, v in self.statisticLabels.items():
            v.destroy()

        self.statisticLabels = {}
        self.statisticLabels['E'] = ttk.Label(self.statisticFrame, text='Среднее: {0}'.format(np.mean(self.y)))
        self.statisticLabels['E'].grid(row=0, sticky='w')
        self.statisticLabels['D'] = ttk.Label(self.statisticFrame, text='Дисперсия: {0}'.format(np.std(self.y)**2))
        self.statisticLabels['D'].grid(row=1, sticky='w')
        self.statisticLabels['sigma'] = ttk.Label(self.statisticFrame, text='Отклонение: {0}'.format(np.std(self.y)))
        self.statisticLabels['sigma'].grid(row=2, sticky='w')
        self.statisticLabels['var'] = ttk.Label(self.statisticFrame, text='Вариация: {0}'.format(np.var(self.y)))
        self.statisticLabels['var'].grid(row=3, sticky='w')
        self.statisticLabels['skew'] = ttk.Label(self.statisticFrame, text='Ассиметрия: {0}'.format(skew(self.y)))
        self.statisticLabels['skew'].grid(row=4, sticky='w')
        self.statisticLabels['kurtosis'] = ttk.Label(self.statisticFrame, text='Экцесс: {0}'.format(kurtosis(self.y)))
        self.statisticLabels['kurtosis'].grid(row=5, sticky='w')
        self.statisticLabels['max'] = ttk.Label(self.statisticFrame, text='Max: {0}'.format(max(self.y)))
        self.statisticLabels['max'].grid(row=6, sticky='w')
        self.statisticLabels['min'] = ttk.Label(self.statisticFrame, text='Min: {0}'.format(min(self.y)))
        self.statisticLabels['min'].grid(row=7, sticky='w')
        self.statisticLabels['median'] = ttk.Label(self.statisticFrame, text='Медиана: {0}'.format(np.median(self.y)))
        self.statisticLabels['median'].grid(row=8, sticky='w')
        self.canvas.draw()
        #self.show_ft()

    def signal(self, t):
        return 0.0

    def dft(self, arr):
        ans = []
        for i in range(len(arr)):
            ans.append(0j)
            for j in range(len(arr)):
                ans[-1] += (arr[j] + 0j) * cmath.exp( -1j * 2 * pi / len(arr) * j * i)

        return ans

    def idft(self, arr):
        ans = []
        for i in range(len(arr)):
            ans.append(0j)
            for j in range(len(arr)):
                ans[-1] += arr[j] * cmath.exp(1j * (2 * pi) / len(arr) * i * j)

            ans[-1] /= len(arr)
        return ans

    def fft(self, arr):
        arr = np.asarray(arr, dtype=float)
        N = arr.shape[0]

        if N % 2 > 0:
            raise ValueError("size of x must be a power of 2")
        elif N <= 32:  # this cutoff should be optimized
            return self.dft(arr)
        else:
            X_even = self.fft(arr[::2])
            X_odd = self.fft(arr[1::2])
            factor = np.exp(-2j * pi * np.arange(N) / N)
            return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                                   X_even + factor[int(N / 2):] * X_odd])

    def ifft(self, arr):
        ans = self.ifft_helper(arr) / len(arr) * 2
        return ans

    def ifft_helper(self, arr):
        arr = np.asarray(arr, dtype=float)
        N = arr.shape[0]

        if N % 2 > 0:
            raise ValueError("size of x must be a power of 2")
        elif N <= 32:  # this cutoff should be optimized
            return self.idft(arr)
        else:
            X_even = self.fft(arr[::2])
            X_odd = self.fft(arr[1::2])
            factor = np.exp(-2j * pi * np.arange(N) / N)
            return np.concatenate([X_even + factor[:int(N / 2)] * X_odd,
                                   X_even + factor[int(N / 2):] * X_odd])

    def toogle_linear(self):
        pass

    def show_ft(self):
        amp_root = AmplitudeWindow([np.arange(0, int(self.deltaT), (self.deltaT) / len(self.x)), abs(self.fft(self.y))])
        #amp_root.title('Amplitude')

        phase_root = tk.Toplevel()
        phase_root.title('Phase')

        #self.draw_amplitude(amp_root)
        self.draw_phase(phase_root)

    def draw_phase(self, root):
        phase = Figure()
        phase.suptitle('Phase')
        phase_canvas = FigureCanvasTkAgg(phase, master=root)
        phase_plot = phase.add_subplot(111)
        phase_plot.plot(self.x, [ cmath.phase(i) for i in self.fft(self.y) ])
        phase_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        phase_canvas.draw()


    def setCallback(self):
        for k, i in self.entries.items():
            self.__dict__[k] = float(i.get())
        self.invalidatePlot()

    def destroy(self):
        self.mainFrame.destroy()

    @staticmethod
    def getName(self):
        return self.name

class DelayedImpulse(BaseSignal):
    n0 = 3
    name = 'Delayed Impulse'
    title = r"$u_0(n-n_0)$"
    def show(self):
        self.initInterface({"t" : self.t, "n0" : self.n0})
        self.basePlot()

    def signal(self, t):
        if t == self.n0:
            return 1
        return 0


class DelayedJump(BaseSignal):
    n0 = 3
    name = 'Delayed Jump'
    title = r'$u_{-1}(n)$'
    def show(self):
        self.initInterface({"t" : self.t, "n0" : self.n0})
        self.basePlot()

    def signal(self, t):
        if t < self.n0:
            return 0
        return 1

class DiscreteDecresingExponent(BaseSignal):
    a = 0.5
    name = 'Discrete Decreasing Exponent'
    title = r'a^n'
    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a})
        self.basePlot()

    def signal(self, t):
        return self.a**t

class DiscreteSinusoid(BaseSignal):
    a = 0.5
    omega = 0.5
    phi = 0.5
    name = 'Discrete Sinusoid'
    title = r'$a \sin{(n \omega + \phi)}$'
    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * sin(t * self.omega + self.phi)

class Meandr(BaseSignal):
    L = 4
    name = 'Meandr'
    title = r'Меандр'
    def show(self):
        self.initInterface({"t" : self.t, "L" : self.L})
        self.basePlot()

    def signal(self, t):
        if (t % self.L) < (self.L / 2):
            return 1
        else :
            return -1

class Saw(BaseSignal):
    L = 4
    name = 'Saw'
    title = r'$\frac{mod(n, L)}{L}$'
    def show(self):
        self.initInterface({"t" : self.t, "L" : self.L})
        self.basePlot()

    def signal(self, t):
        return (t % self.L) / self.L

class ExponentЕnvelope(BaseSignal):
    a = 2
    tau = 3
    omega = 0.5
    phi = 1.5
    name = 'Exponent Еnvelope'
    title = r'$a e^{-\frac{t}{\tau}} \cos{(\omega t + \phi)}$'
    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "tau" : self.tau,
                            "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * exp(-t/self.tau)*cos(self.omega*t+self.phi)

class BalanceEnvelope(BaseSignal):
    a = 2
    u = 1.5
    omega = 0.5
    phi = 1.5
    name = 'Balance Envelope'
    title = r'$a \cos{(u t)} \cos{(\omega t + \phi)}$'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)

    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "u" : self.u,
                            "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * cos(self.u * t)* cos(self.omega * t + self.phi)

class TonalEnvelope(BaseSignal):
    a = 2.1
    m = 0.3
    u = 0.5
    omega = 1.5
    phi = 0.5
    name = 'Tonal Envelope'
    title = r'$a(1 + m \cos{(u t)}) \cos{(\omega t + \phi)}$'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)

    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "m" : self.m,
                            "u" : self.u, "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * (1 + self.m * cos(self.u * t) * cos(self.omega * t + self.phi))

class WhiteNoise(BaseSignal):
    a = 0
    b = 1
    name = "White Noise"
    title = name
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)

    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "b" : self.b})
        self.basePlot()

    def signal(self, t):
        return random.uniform(self.a, self.b)

class NormalNoise(BaseSignal):
    a = 0
    d = 1
    name = "Normal Noise"
    title = name
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)

    def show(self):
        self.initInterface({"t" : self.t, "a" : self.a, "d" : self.d})
        self.basePlot()

    def signal(self, t):
        return random.gauss(self.a, self.d)

class Signals(tk.Tk):
    signals = ( DelayedImpulse, DelayedJump, DiscreteDecresingExponent,
                DiscreteSinusoid, Meandr, Saw, ExponentЕnvelope, BalanceEnvelope,
                TonalEnvelope, WhiteNoise, NormalNoise )

    def show_frame(self, cont):
        if self.fr != None:
            self.fr.destroy()
        self.fr = self.frames[cont]
        self.fr.show()
        #frame.tkraise()

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Lab4")
        self.geometry("640x527")

        self.fr = None #??

        self.container = tk.Frame(self)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)    #?
        self.container.grid_columnconfigure(0, weight=1) #?

        menubar = tk.Menu(self)
        signalMenu = tk.Menu(menubar, tearoff=0)

        c = 0
        self.frames = {}
        for F in self.signals:
            frame = F(self.container, self)
            #frame.grid(row=0, column=0, sticky="news")
            self.frames[str(c)] = frame
            signalMenu.add_command(label=F.getName(F), command=(self.register(lambda x : self.show_frame(x)), c))
            c += 1

        menubar.add_cascade(label='Signals', menu=signalMenu)
        self.config(menu=menubar)
        self.show_frame('0') #todo fix

app = Signals()
app.mainloop()
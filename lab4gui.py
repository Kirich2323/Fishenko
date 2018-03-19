import tkinter as tk
from tkinter import ttk
import matplotlib
import re
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from math import sin, cos, exp

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

class BaseSignal(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.controller = controller
        self.vcmd = (self.parent.register(digits_entry_check), "%d", "%i", "%P", "%s", "%S", "%v", "%V", "%V")
        self.t = 10
        self.deltaT = 1.0

    def initInterface(self, properties):
        propertyFrame = ttk.Frame(self, width=160, height=100)
        labels = {}
        self.entries = {}
        c = 0
        for k, i in properties.items():
            labels[k] = ttk.Label(propertyFrame, text=k)
            labels[k].grid(row=c, column=0)
            self.entries[k] = ttk.Entry(propertyFrame, validate="key", validatecommand=self.vcmd, width=7)
            self.entries[k].insert(0, str(i))
            self.entries[k].grid(row=c, column=1)
            c += 1

        setButton = ttk.Button(propertyFrame, text="Set", width=7, command=self.setCallback)
        setButton.grid(row=len(labels), column=1, sticky="WE")
        propertyFrame.pack(side=tk.LEFT)

    def basePlot(self):
        self.figure = Figure(figsize=(5,5), dpi=100) #check settings
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.plot = self.figure.add_subplot(111)
        self.invalidatePlot()
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)


        toolbar = NavigationToolbar2TkAgg(self.canvas, self)
        toolbar.update()

    def invalidatePlot(self):
        self.plot.clear()
        x = [ i * self.deltaT for i in range(int(self.t)) ]
        y = [ self.signal(i) for i in x ]
        self.plot.plot(x, y)
        self.canvas.draw()

    def signal(self, t):
        return 0.0

    def setCallback(self):
        for k, i in self.entries.items():
            self.__dict__[k] = float(i.get())
        self.invalidatePlot()

    @staticmethod
    def getName(self):
        return self.name

class DelayedImpulse(BaseSignal):
    n0 = 3
    name = 'Delayed Impulse'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "n0" : self.n0})
        self.basePlot()

    def signal(self, t):
        if t == self.n0:
            return 1
        return 0


class DelayedJump(BaseSignal):
    n0 = 3
    name = 'Delayed Jump'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "n0" : self.n0})
        self.basePlot()

    def signal(self, t):
        if t < self.n0:
            return 0
        return 1

class DiscreteDecresingExponent(BaseSignal):
    a = 0.5
    name = 'Discrete Decreasing Exponent'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "a" : self.a})
        self.basePlot()

    def signal(self, t):
        return self.a**t

class DiscreteSinusoid(BaseSignal):
    a = 0.5
    omega = 0.5
    phi = 0.5
    name = 'Discrete Sinusoid'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "a" : self.a, "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * sin(t * self.omega + self.phi)

class Meandr(BaseSignal):
    L = 4
    name = 'Meandr'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
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
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "L" : self.L})
        self.basePlot()

    def signal(self, t):
        return (t % self.L) / self.L

class Saw(BaseSignal):
    L = 4
    name = 'Saw'
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
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
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
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
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
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
    def __init__(self, parent, controller):
        BaseSignal.__init__(self, parent, controller)
        self.initInterface({"t" : self.t, "a" : self.a, "m" : self.m,
                            "u" : self.u, "omega" : self.omega, "phi" : self.phi})
        self.basePlot()

    def signal(self, t):
        return self.a * (1 + self.m * cos(self.u * t) * cos(self.omega * t + self.phi))

class Signals(tk.Tk):
    signals = ( DelayedImpulse, DelayedJump, DiscreteDecresingExponent,
                DiscreteSinusoid, Meandr, Saw, ExponentЕnvelope, BalanceEnvelope, TonalEnvelope )

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Lab4")
        self.geometry("640x527")

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
            frame.grid(row=0, column=0, sticky="news")
            self.frames[str(c)] = frame
            signalMenu.add_command(label=F.getName(F), command=(self.register(lambda x : self.show_frame(x)), c))
            c += 1

        menubar.add_cascade(label='Signals', menu=signalMenu)
        self.config(menu=menubar)
        self.show_frame('0') #todo fix

app = Signals()
app.mainloop()
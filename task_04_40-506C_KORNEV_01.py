import numpy
import tools
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    #Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 4500

    #Размер области моделирования в метрах
    X = 1.0

    #Размер ячейки разбиения
    dx = 1e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = 50

    # Параметры модулированного гаусса
    A0 = 100        #Ослабление сигнала в момент времени t=0
    Amax = 100      #Уровень ослабления спектра сигнала на частоте Fmax
    DF = 3e9        #Ширина спектра по уровню ослабления Amax
    f0 = 2.5e9      #Центральная частота в спектре сигнала
    Wg = 2 * numpy.sqrt(numpy.log(Amax)) / (numpy.pi * DF)
    Dg = Wg * numpy.sqrt(numpy.log(A0))
    Nwg = Wg / dt
    Ndg = Dg / dt
    N1 = Sc / (f0 * dt)

    # Датчики для регистрации поля
    probesPos = [25,75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    #1й слой диэлектрика
    eps1 = 4
    d1 = 0.1
    layer_1 = int(maxSize / 2) + int(d1 / dx)

    #2й слой диэлектрика
    eps2 = 9
    d2 = 0.05
    layer_2 = layer_1 + int(d2 / dx)

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[int(maxSize/2):layer_1] = eps1
    eps[layer_1:layer_2+1] = eps2
    
    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])
    display.drawBoundary(int(maxSize / 2))
    display.drawBoundary(layer_1)
    display.drawBoundary(layer_2)

    for t in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / (W0 * mu[sourcePos - 1]) *
                              numpy.sin(2 * numpy.pi * t * Sc / N1) *
                              numpy.exp(-(t - Ndg) ** 2 / (Nwg ** 2)))

        # Граничные условия для поля E
        Ez[0] = Ez[1]
        Ez[-1] = Ez[-2]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          numpy.sin(2 * numpy.pi / N1 * ((t + 0.5) * Sc -
                          (-0.5 * numpy.sqrt(eps[sourcePos] * mu[sourcePos])))) *
                          numpy.exp(-(t + 0.5 - (-0.5 * numpy.sqrt(eps[sourcePos] *
                          mu[sourcePos]) / Sc) - Ndg) ** 2 / (Nwg ** 2)))

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 20 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    #Спектр сигнала в датчиках
    #Размер массива
    size = 2 ** 16

    #Шаг по частоте
    df = 1.0 / (size * dt)
    f = numpy.arange(- size / 2 * df, size / 2 * df, df)
        
    #Расчет спектра падающего поля
    fall = numpy.zeros(maxTime)
    fall[:int(0.25 * 1e-8 /dt)] = probes[1].E[:int(0.25 * 1e-8 /dt)]
    spectrum_fall = numpy.abs(fft(fall, size))
    spectrum_fall = fftshift(spectrum_fall)
    
    #Расчет спектра отраженного поля
    spectrum_reflected = numpy.abs(fft(probes[0].E, size))
    spectrum_reflected = fftshift(spectrum_reflected)
    
    #Построение графиков
    plt.plot(f, spectrum_fall /numpy.max(spectrum_fall))
    plt.plot(f, spectrum_reflected /numpy.max(spectrum_reflected))
    plt.title('Спектральная плотность энергии падающего и отраженного сигналов')
    plt.grid()
    plt.xlim(0, 5e9)
    plt.xlabel('f, Гц')
    plt.ylabel(r'$\frac{|S|}{S_{max}}$')
    plt.legend(['СПЭ пад. сигнала','СПЭ отр. сигнала'], loc = 1)
    plt.show()

    plt.figure()
    plt.plot(f, spectrum_reflected / spectrum_fall)
    plt.grid()
    plt.title('Зависимость модуля коэффициента отражения от частоты')
    plt.xlim(1e9, 4e9)
    plt.ylim(-0.05, 0.85)
    plt.xlabel('f, Гц')
    plt.ylabel('|Г|')
    plt.show()

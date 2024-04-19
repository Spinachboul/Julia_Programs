using StaticArrays
using DifferentialEquations

module BeamProblem

    const N = 40
    const NN = 2 * N
    const NCOM = N
    const NSQ = N * N
    const NQUATR = NSQ * NSQ
    const DELTAS = 1.0 / AN

    function pidate()
        return 20060828
    end

    function prob(fullnm, problm, type, neqn, ndisc, t, numjac, mljac, mujac, nummas, mlmas, mumas, ind)
        fullnm = "Beam"
        problm = "beam"
        type = "ODE"
        neqn = 80
        ndisc = 0
        t[0] = 0.0
        t[1] = 5.0
        numjac = true
        mljac = neqn
        mujac = neqn
        return nothing
    end

    function init(neqn, t, y, yprime, consis)
        y .= 0.0
        return nothing
    end

    function settolerances(neqn, rtol, atol, tolvec)
        tolvec[1:neqn] .= false
        return nothing
    end

    function setoutput(neqn, solref, printsolout, nindsol, indsol)
        solref = true
        printsolout = true
        nindsol = 4
        indsol[1] = 10
        indsol[2] = 20
        indsol[3] = 30
        indsol[4] = 40
        return nothing
    end

    function feval(nn, t, th, yprime, df, ierr, rpar, ipar)
        df = zeros(nn)
        th = zeros(150)
        u = zeros(150)
        v = zeros(150)
        w = zeros(150)
        alpha = zeros(150)
        beta = zeros(150)
        sth = zeros(150)
        cth = zeros(150)
        if t > π
            # Case t greater than pi
            term1 = (-3.0 * th[1] + th[2]) * NQUATR
            v[1] = term1
            for i = 2:N-1
                term1 = (th[i-1] - 2.0 * th[i] + th[i+1]) * NQUATR
                v[i] = term1
            end
            term1 = (th[N-1] - th[N]) * NQUATR
            v[N] = term1
        else
            # Case t less than or equal to pi
            fabs = 1.5 * sin(t) * sin(t)
            fx = -fabs
            fy = fabs
            term1 = (-3.0 * th[1] + th[2]) * NQUATR
            term2 = NSQ * (fy * cos(th[1]) - fx * sin(th[1]))
            v[1] = term1 + term2
            for i = 2:N-1
                term1 = (th[i-1] - 2.0 * th[i] + th[i+1]) * NQUATR
                term2 = NSQ * (fy * cos(th[i]) - fx * sin(th[i]))
                v[i] = term1 + term2
            end
            term1 = (th[N-1] - th[N]) * NQUATR
            term2 = NSQ * (fy * cos(th[N]) - fx * sin(th[N]))
            v[N] = term1 + term2
        end
        w[1] = sth[2] * v[2]
        for i = 2:N-1
            w[i] = -sth[i] * v[i-1] + sth[i+1] * v[i+1]
        end
        w[N] = -sth[N] * v[N-1]
        for i = 1:N
            w[i] += th[N+i] * th[N+i]
        end
        alpha[1] = 1.0
        for i = 2:N
            alpha[i] = 2.0
        end
        alpha[N] = 3.0
        for i = N-1:-1:1
            q = beta[i] / alpha[i+1]
            w[i] -= w[i+1] * q
            alpha[i] -= beta[i] * q
        end
        w[1] /= alpha[1]
        for i = 2:N
            w[i] = (w[i] - beta[i-1] * w[i-1]) / alpha[i]
        end
        u[1] = v[1] - cth[2] * v[2] + sth[2] * w[2]
        for i = 2:N-1
            u[i] = 2.0 * v[i] - cth[i] * v[i-1] - cth[i+1] * v[i+1] - sth[i] * w[i-1] + sth[i+1] * w[i+1]
        end
        u[N] = 3.0 * v[N] - cth[N] * v[N-1] - sth[N] * w[N-1]
        df[1:N] .= th[N+1:2*N]
        df[N+1:2*N] .= u[1:N]
        return nothing
    end

    function jeval(ldim, neqn, t, y, yprime, dfdy, ierr, rpar, ipar)
        return nothing  # Dummy subroutine
    end

    function meval(ldim, neqn, t, y, yprime, dfddy, ierr, rpar, ipar)
        return nothing  # Dummy subroutine
    end

    function solut(neqn, t, y)
        y .= [
            -0.5792366591285007, -0.1695298550721735, -0.2769103312973094, -0.3800815655879158, -0.4790616859743763,
            -0.5738710435274594, -0.6645327313454617, -0.7510730581979037, -0.8335219765414992, -0.9119134654647947,
            -0.9862858700132091, -0.1056682200378002, -0.1123150395409595, -0.1185743552727245, -0.1244520128755561,
            -0.1299544113264161, -0.1350885180610398, -0.1398618819194422, -0.1442826441015242, -0.1483595472463012,
            -0.1521019429001447, -0.1555197978061129, -0.1586236993420229, -0.1614248603702127, -0.1639351238193223,
            -0.1661669673440852, -0.1681335081778558, -0.1698485080602243, -0.1713263782440888, -0.1725821847462537,
            -0.1736316537975654, -0.1744911773840049, -0.1751778187863392, -0.1757093178712902, -0.1761040960228576,
            -0.1763812607175549, -0.1765606097564671, -0.1766626352260565, -0.1767085270807460, -0.1767201761075488,
            0.3747362681329794, 0.1099117880217593, 0.1798360474312799, 0.2472427305442391, 0.3121293820596567,
            0.3744947377019500, 0.4343386073492798, 0.4916620354601748, 0.5464677854586807, 0.5987609702624270,
            0.6485493611110740, 0.6958435169132503, 0.7406572668520808, 0.7830081747813241, 0.8229176659201515,
            0.8604110305190560, 0.8955175502377805, 0.9282708263127953, 0.9587089334522034, 0.9868747821728363,
            0.1012816579961883, 0.1036587736679858, 0.1058246826481355, 0.1077857811433353, 0.1095490222005369,
            0.1111219164294898, 0.1125125269286501, 0.1137294526609229, 0.1147818025153607, 0.1156792132004482,
            0.1164318845130183, 0.1170505992596124, 0.1175467424299550, 0.1179323003228859, 0.1182198586299667,
            0.1184226111223146, 0.1185543909805575, 0.1186297084203716, 0.1186637618908127, 0.1186724615113034
        ]
        return nothing
    end

end

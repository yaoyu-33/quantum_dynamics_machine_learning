/*******************************************************************************
 * Quantum dynamics (QD) simulation of an electron in one dimension.
 *
 * USAGE
 *
 * %cc -o qd1 qd1.c -lm
 * %qd1 < qd1.in (see qd1.h for the input-file format)
 * *******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "qd1.h"


int main(int argc, char *argv[]) {
    int i, step; /* Simulation loop iteration index */
    double t,r,x;
    FILE *fp;

    init_param();  /* Read input parameters */
    if (argc >= 10) {
        X0 = atof(argv[1]);
        S0 = atof(argv[2]);
        E0 = atof(argv[3]);
        fp = fopen(argv[4], "w+");

        /* Set up potential */
        for (i=1; i<=NX; i++) {
            v[i] = atof(argv[4+i]);
        }
    }
    else {
        X0 = atof(argv[1]);
        S0 = atof(argv[2]);
        E0 = atof(argv[3]);
        BH = atof(argv[4]);
        BW = atof(argv[5]);
        EH = atof(argv[6]);
        fp = fopen(argv[7], "w+");

        /* Set up potential */
        for (i=1; i<=NX; i++) {
            x = dx*i;
            /* Construct the edge potential */
            if (i==1 || i==NX)
                v[i] = EH;
            /* Construct the barrier potential */
            else if (0.5*(LX-BW)<x && x<0.5*(LX+BW))
                v[i] = BH;
            else
                v[i] = 0.0;
        }
    }


    init_prop();   /* Initialize the kinetic & potential propagators */
    init_wavefn(); /* Initialize the electron wave function */

    for (step=0; step<=NSTEP; step++) {
        if (step % 200 == 0) {
            fprintf(fp, "timestamp: %.5f\n", step * DT);
            fprintf(fp, "params: %d %lf %lf %lf %lf %lf %lf %lf %lf\n", NX, LX, DT, X0, S0, E0, BH, BW, EH);
            fprintf(fp, "psi_re: ");
            for (i = 1; i <= NX; i++) {
                fprintf(fp, "%le ", psi[i][0]);
            }
            fprintf(fp, "\n");
            fprintf(fp, "psi_im: ");
            for (i = 1; i <= NX; i++) {
                fprintf(fp, "%le ", psi[i][1]);
            }
            fprintf(fp, "\n");
            fprintf(fp, "pot: ");
            for (i = 1; i <= NX; i++) {
                fprintf(fp, "%le ", v[i]);
            }
            fprintf(fp, "\n\n");
        }

        single_step(); /* Time propagation for one step, DT */
    }
    fclose(fp);
    return 0;
}

/*----------------------------------------------------------------------------*/
void init_param() {
/*------------------------------------------------------------------------------
 *     Initializes parameters by reading them from standard input.
 *     ------------------------------------------------------------------------------*/
    /* Read control parameters */
    LX =  100.0;
    DT = 0.0005;
    NSTEP = 100000;
    NECAL = 1000;
    X0 = 12.5;
    S0 = 1.0;
    E0 = 10.0;
    BH = 0.0;
    BW = 4.0;
    EH = 0.0;

    /* Calculate the mesh size */
    dx = LX/NX;
}

/*----------------------------------------------------------------------------*/
void init_prop() {
/*------------------------------------------------------------------------------
 *     Initializes the kinetic & potential propagators.
 *     ------------------------------------------------------------------------------*/
    int stp,s,i,up,lw;
    double a,exp_p[2],ep[2],em[2];
    double x;

    /* Set up kinetic propagators */
    a = 0.5/(dx*dx);

    for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
        exp_p[0] = cos(-(stp+1)*DT*a);
        exp_p[1] = sin(-(stp+1)*DT*a);
        ep[0] = 0.5*(1.0+exp_p[0]);
        ep[1] = 0.5*exp_p[1];
        em[0] = 0.5*(1.0-exp_p[0]);
        em[1] = -0.5*exp_p[1];

        /* Diagonal propagator */
        for (s=0; s<2; s++) al[stp][s] = ep[s];

        /* Upper & lower subdiagonal propagators */
        for (i=1; i<=NX; i++) { /* Loop over mesh points */
            if (stp==0) { /* Half-step */
                up = i%2;     /* Odd mesh point has upper off-diagonal */
                lw = (i+1)%2; /* Even               lower              */
            }
            else { /* Full step */
                up = (i+1)%2; /* Even mesh point has upper off-diagonal */
                lw = i%2;     /* Odd                 lower              */
            }
            for (s=0; s<2; s++) {
                bux[stp][i][s] = up*em[s];
                blx[stp][i][s] = lw*em[s];
            }
        } /* Endfor mesh points, i */
    } /* Endfor half & full steps, stp */

    /* Set up potential propagator */
    for (i=1; i<=NX; i++) {
        /* Half-step potential propagator */
        u[i][0] = cos(-0.5*DT*v[i]);
        u[i][1] = sin(-0.5*DT*v[i]);
    }
}

/*----------------------------------------------------------------------------*/
void init_wavefn() {
/*------------------------------------------------------------------------------
 *     Initializes the wave function as a traveling Gaussian wave packet.
 *     ------------------------------------------------------------------------------*/
    int sx,s;
    double x,gauss,psisq,norm_fac;

    /* Calculate the the wave function value mesh point-by-point */
    for (sx=1; sx<=NX; sx++) {
        x = dx*sx-X0;
        gauss = exp(-0.25*x*x/(S0*S0));
        // if (fabs(x) < S0)
        //     gauss = S0-fabs(x);
        // else
        //     gauss = 0.0;
        psi[sx][0] = gauss*cos(sqrt(2.0*E0)*x);
        psi[sx][1] = gauss*sin(sqrt(2.0*E0)*x);
    }

    /* Normalize the wave function */
    psisq=0.0;
    for (sx=1; sx<=NX; sx++)
        for (s=0; s<2; s++)
            psisq += psi[sx][s]*psi[sx][s];
    psisq *= dx;
    norm_fac = 1.0/sqrt(psisq);
    for (sx=1; sx<=NX; sx++)
        for (s=0; s<2; s++)
            psi[sx][s] *= norm_fac;
}

/*----------------------------------------------------------------------------*/
void single_step() {
/*------------------------------------------------------------------------------
 *     Propagates the electron wave function for a unit time step, DT.
 *     ------------------------------------------------------------------------------*/
    pot_prop();  /* half step potential propagation */

    kin_prop(0); /* half step kinetic propagation   */
    kin_prop(1); /* full                            */
    kin_prop(0); /* half                            */

    pot_prop();  /* half step potential propagation */
}

/*----------------------------------------------------------------------------*/
void pot_prop() {
/*------------------------------------------------------------------------------
 *     Potential propagator for a half time step, DT/2.
 *     ------------------------------------------------------------------------------*/
    int sx;
    double wr,wi;

    for (sx=1; sx<=NX; sx++) {
        wr=u[sx][0]*psi[sx][0]-u[sx][1]*psi[sx][1];
        wi=u[sx][0]*psi[sx][1]+u[sx][1]*psi[sx][0];
        psi[sx][0]=wr;
        psi[sx][1]=wi;
    }
}

/*----------------------------------------------------------------------------*/
void kin_prop(int t) {
/*------------------------------------------------------------------------------
 *     Kinetic propagation for t (=0 for DT/2--half; 1 for DT--full) step.
 *     -------------------------------------------------------------------------------*/
    int sx,s;
    double wr,wi;

    /* Apply the periodic boundary condition */
    periodic_bc();

    /* WRK|PSI holds the new|old wave function */
    for (sx=1; sx<=NX; sx++) {
        wr = al[t][0]*psi[sx][0]-al[t][1]*psi[sx][1];
        wi = al[t][0]*psi[sx][1]+al[t][1]*psi[sx][0];
        wr += (blx[t][sx][0]*psi[sx-1][0]-blx[t][sx][1]*psi[sx-1][1]);
        wi += (blx[t][sx][0]*psi[sx-1][1]+blx[t][sx][1]*psi[sx-1][0]);
        wr += (bux[t][sx][0]*psi[sx+1][0]-bux[t][sx][1]*psi[sx+1][1]);
        wi += (bux[t][sx][0]*psi[sx+1][1]+bux[t][sx][1]*psi[sx+1][0]);
        wrk[sx][0] = wr;
        wrk[sx][1] = wi;
    }

    /* Copy the new wave function back to PSI */
    for (sx=1; sx<=NX; sx++)
        for (s=0; s<=1; s++)
            psi[sx][s] = wrk[sx][s];
}

/*----------------------------------------------------------------------------*/
void periodic_bc() {
/*------------------------------------------------------------------------------
 *     Applies the periodic boundary condition to wave function PSI, by copying
 *         the boundary values to the auxiliary array positions at the other ends.
 *         ------------------------------------------------------------------------------*/
    int s;

    /* Copy boundary wave function values */

    for (s=0; s<=1; s++) {
        psi[0][s] = psi[NX][s];
        psi[NX+1][s] = psi[1][s];
    }
}

/*----------------------------------------------------------------------------*/
void calc_energy() {
/*------------------------------------------------------------------------------
 *     Calculates the kinetic, potential & total energies, EKIN, EPOT & ETOT.
 *     ------------------------------------------------------------------------------*/
    int sx,s;
    double a,bx;

    /* Apply the periodic boundary condition */

    /* Tridiagonal kinetic-energy operators */
    a =   1.0/(dx*dx);
    bx = -0.5/(dx*dx);

    /* |WRK> = (-1/2)Laplacian|PSI> */
    for (sx=1; sx<=NX; sx++)
        for (s=0; s<=1; s++)
            wrk[sx][s] = a*psi[sx][s]+bx*(psi[sx-1][s]+psi[sx+1][s]);

    /* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
    ekin = 0.0;
    for (sx=1; sx<=NX; sx++)
        ekin += (psi[sx][0]*wrk[sx][0]+psi[sx][1]*wrk[sx][1]);
    ekin *= dx;

    /* Potential energy */
    epot = 0.0;
    for (sx=1; sx<=NX; sx++)
        epot += v[sx]*(psi[sx][0]*psi[sx][0]+psi[sx][1]*psi[sx][1]);
    epot *= dx;

    /* Total energy */
    etot = ekin+epot;
}

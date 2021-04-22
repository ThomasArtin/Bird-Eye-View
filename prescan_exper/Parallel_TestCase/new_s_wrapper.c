
/*
 * Include Files
 *
 */
#if defined(MATLAB_MEX_FILE)
#include "tmwtypes.h"
#include "simstruc_types.h"
#else
#include "rtwtypes.h"
#endif



/* %%%-SFUNWIZ_wrapper_includes_Changes_BEGIN --- EDIT HERE TO _END */
#include <math.h>
/* %%%-SFUNWIZ_wrapper_includes_Changes_END --- EDIT HERE TO _BEGIN */
#define u_width 1
#define y_width 1

/*
 * Create external references here.  
 *
 */
/* %%%-SFUNWIZ_wrapper_externs_Changes_BEGIN --- EDIT HERE TO _END */
/* extern double func(double a); */
/* %%%-SFUNWIZ_wrapper_externs_Changes_END --- EDIT HERE TO _BEGIN */

/*
 * Output functions
 *
 */
void new_s_Outputs_wrapper(const real_T *Front,
			const real_T *LEFT,
			const real_T *RIGHT,
			const real_T *BACK,
			const real_T *current_v,
			const real_T *X,
			const real_T *Y,
			real_T *steering,
			real_T *velocity,
			real_T *brake,
			real_T *Trottle,
			real_T *counter,
			real_T *counter1,
			real_T *counter2,
			real_T *counter3,
			real_T *autog)
{
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_BEGIN --- EDIT HERE TO _END */

if (RIGHT != 0 && counter3 == 0){
    *velocity = 5;
    *steering = 0;
}
else{
    counter3 =1 ;
    *brake = 150;
    *velocity = 0;
    
}
/* %%%-SFUNWIZ_wrapper_Outputs_Changes_END --- EDIT HERE TO _BEGIN */
}



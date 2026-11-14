package net.montaki.neuralnetworkdesignpatterns.aspects;

import java.util.Arrays;
import net.montaki.neuralnetworkdesignpatterns.security.RequiresRole;
import net.montaki.neuralnetworkdesignpatterns.security.SecurityContext;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.Around;
import org.aspectj.lang.annotation.Aspect;

@Aspect
public class AuthorizationAspect {
    @Around(value = "execution(* *(..)) && @annotation(permission)", argNames = "joinPoint,permission")
    public Object authorize(ProceedingJoinPoint joinPoint, RequiresRole permission) throws Throwable {
        if (!SecurityContext.isAuthenticated()) {
            throw new SecurityException("Authentification requise");
        }
        if (Arrays.stream(permission.value()).noneMatch(SecurityContext::hasRole)) {
            throw new SecurityException("Acces refuse a " + joinPoint.getSignature().toShortString());
        }
        return joinPoint.proceed();
    }
}
